# f2_tx_path class 
# Last modification by Marko Kosunen, marko.kosunen@aalto.fi, 18.11.2018 13:51
# This is the description of the DSP for a single TX path in a MIMO transmitter
import numpy as np
import scipy.signal as sig
import tempfile
import subprocess
import shlex
import time

from thesdk import *
from verilog import *
from f2_util_classes import *
from f2_interpolator import *
import signal_generator_802_11n as sg80211n


class f2_tx_path(verilog,thesdk):
    @property
    def _classfile(self):
        return os.path.dirname(os.path.realpath(__file__)) + "/"+__name__

    def __init__(self,*arg): 
        self.proplist = [ 'Rs', 
                          'Rs_dsp', 
                          'Txbits', 
                          'Users', 
                          'interpolator_scales' 
                          'interpolator_cic3shift' 
                          ];                    #properties that can be 
                                                #propagated from parent
        self.Rs = 160e6;                        # Highes sampling frequency
        self.Rs_dsp=20e6

        #These are fixed
        self.Txbits=9                          # Bits of DAC
        ####
        self.Users=4                           #This is currently fixed by implementation
        self.interpolator_scales=[8,2,2,512]   #This works with the current hardware
        self.interpolator_cic3shift=4
        self.user_sum_mode    = 0              #Wether to sum users or not
        self.user_select_index= 0              #by default, no parallel processing
        #self.user_delays      = [ 0 for i in range(self.Users)] 
        self.interpolator_mode= 4
        self.dac_data_mode    = 6
        self.bin              = 4
        #Matrix of [Users,time,1]
        self.model='py';                  #can be set externally, but is not propagated
        self.par= False                   #by default, no parallel processing

        self.queue= []                    #by default, no parallel processing
        self.DEBUG= False
        if len(arg)>=1:
            parent=arg[0]
            self.copy_propval(parent,self.proplist)
            self.parent =parent;

        self.iptr_A =[ refptr for i in range(self.Users) ] 
        self.user_weights      = [ (1+0j) for i in range(self.Users) ]
        #Thermometer and binary weighted outputs
        self._Z_real_t= refptr()
        self._Z_real_b= refptr()
        self._Z_imag_t= refptr()
        self._Z_imag_b= refptr()

        self.init()
    
    def init(self):
        self.interpolator=f2_interpolator(self)
        self.interpolator.Rs_high=self.Rs
        self.interpolator.Rs_low =self.Rs_dsp
        self.interpolator.init()
        #Get the interpolator mode from the interpolator
        self.interpolator_mode=self.interpolator.mode

        ##Here's how we sim't for the tapeout
        #self._vlogparameters=dict([ ('g_Rs_high',self.Rs), ('g_Rs_low',self.Rs_dsp), 
        #    ('g_shift'            , 0                          ),
        #    ('g_scale0'           , self.interpolator_scales[0]),
        #    ('g_scale1'           , self.interpolator_scales[1]),
        #    ('g_scale2'           , self.interpolator_scales[2]),
        #    ('g_scale3'           , self.interpolator_scales[3]),
        #    ('g_cic3shift'        , self.interpolator_cic3shift),
        #    ('g_user_spread_mode' , self.user_spread_mode      ),
        #    ('g_user_sum_mode'    , self.user_sum_mode         ), 
        #    ('g_user_select_index', self.user_select_index     ),
        #    ('g_interpolator_mode', self.interpolator_mode     ),
        #    ('g_dac_data_mode'    , self.dac_data_mode         ) 
        #    ])
        
    def run(self,*arg):
        if len(arg)>0:
            self.par=True      #flag for parallel processing
            self.queue=arg[0]  #multiprocessing.queue as the first argument
        if self.model=='py':
            self.process_input()
        elif self.model=='sv':
            self.print_log({'type':'F','msg':'Verilog model not available'})
            #self.write_infile()
            ##Object to have handle for it in other methods
            ##Should be handled with a selector method using 'file' attribute
            #a=verilog_iofile(self,**{'name':'Z'})
            #a.simparam='-g g_outfile='+a.file
            #self.run_verilog()
            #self.read_outfile()
            #[ _.remove() for _ in self.iofiles ]

    def process_input(self):
        weighted_users=[ self.iptr_A[i].Value*self.user_weights[i] for i in range(self.Users) ]
        if self.user_sum_mode==1:
            userssum=reduce(lambda prev, next: prev+next,weighted_users )
        else:
            userssum=weighted_users[self.user_select_index]
        #Process the data
        self.interpolator.iptr_A.Value=userssum
        self.interpolator.run()
        self.segment_output()

    #Save these for future verilog sim
    #def write_infile(self,**kwargs):
    #    for i in range(self.Users):
    #        if i==0:
    #            indata=self.iptr_A.data[i].udata.Value.reshape(-1,1)
    #        else:
    #            indata=np.r_['1',indata,self.iptr_A.data[i].udata.Value.reshape(-1,1)]
    #    if self.model=='sv':
    #        #This adds an iofile to self.iiofiles list
    #        a=verilog_iofile(self,**{'name':'A','data':indata})
    #        print(self.iofiles)
    #        a.simparam='-g g_infile='+a.file
    #        a.write()
    #        indata=None #Clear variable to save memory
    #    else:
    #        pass

    #def read_outfile(self):
    #    #Handle the ofiles here as you see the best
    #    a=list(filter(lambda x:x.name=='Z',self.iofiles))[0]
    #    a.read(**{'dtype':'object'})
    #    for i in range(self.Txantennas):
    #        self._Z_real_t[i].Value=a.data[:,i*self.Txantennas+0].astype('str').reshape(-1,1)
    #        self._Z_real_b[i].Value=a.data[:,i*self.Txantennas+1].astype('int').reshape(-1,1)
    #        self._Z_imag_t[i].Value=a.data[:,i*self.Txantennas+2].astype('str').reshape(-1,1)
    #        self._Z_imag_b[i].Value=a.data[:,i*self.Txantennas+3].astype('int').reshape(-1,1)

    def segment_output(self):
        self.print_log({'type':'I',
                'msg':'Normalizing output to %s bits before segmentation'%(self.Txbits)})
        max=np.amax(np.maximum(np.abs(np.real(self.interpolator._Z.Value))
            ,np.abs(np.imag(self.interpolator._Z.Value))))
        # Normalize, shift to positive and quantize
        norm=np.round((self.interpolator._Z.Value/max+(1+1j))/2*(2**self.Txbits-1))

        #Convert thermo to string
        real_t=np.floor(np.real(norm)/(2**self.bin)).astype('int') 
        imag_t=np.floor(np.imag(norm)/(2**self.bin)).astype('int')
        self._Z_real_b.Value=np.remainder(np.real(norm),2**self.bin).astype('int')
        self._Z_imag_b.Value=np.remainder(np.imag(norm),2**self.bin).astype('int')

        # Segment the thermopart 
        segment=np.zeros((len(real_t),2**(self.Txbits-self.bin)-1)).astype('int')
        for i in range(len(real_t)):
            if real_t[i,0] > 0:
                segment[i,-real_t[i,0]:]=np.ones((1,real_t[i,0])).astype('int')

        self._Z_real_t.Value=np.sum(segment.astype('str').astype(np.object),axis=1).astype('str').reshape(-1,1)

        segment=np.zeros((len(real_t),2**(self.Txbits-self.bin)-1)).astype('int')
        for i in range(len(imag_t)):
            if imag_t[i,0] > 0:
                segment[i,-imag_t[i,0]:]=np.ones((1,imag_t[i,0])).astype('int')
        self._Z_imag_t.Value=np.sum(segment.astype('str').astype(np.object),axis=1).astype('str').reshape(-1,1)
        del segment
        del real_t
        del imag_t



