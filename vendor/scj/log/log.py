import logging
import datetime
import os
import time

global show
show = True

def set_up(filename, show_):
    global show
    show = show_
    if show==True:
        print('Set up log '+ str(os.getcwd()).split('run')[0]+f'run\log\{filename}')
    logging.basicConfig(filename=str(os.getcwd()).split('run')[0]+f'run\log\{filename}', filemode='a', level=logging.INFO)
    logging.info('\n')
    logging.info('--------------------------------------------------------------------------------')
    logging.info('--------------'+str(datetime.datetime.now())+'-Start log '+filename+'------------')

def write_log(text):
    global show
    if show == True:
        print('write log:  '+ str(text))
    logging.info(str(time.localtime().tm_hour)+'.'+str(time.localtime().tm_min)+'- '+str(text))

class Write_text:
    def __init__(self, filename, mode='a'):
        self.filename = filename
        self.f = open(self.filename, mode= mode)
        self.f.write("***********************************************************************************************************************"'\n')
        self.f.close()

    def write(self, text):
        print(text)
        self.f = open(self.filename, "a")
        self.f.write(str(text)+'\n')
        self.f.close()

    def write_deal(self, deal= []):
        print(f'Write {deal}')
