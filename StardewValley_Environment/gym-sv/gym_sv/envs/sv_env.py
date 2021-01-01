# part taken from adborghi fantastic implementation
# https://github.com/aborghi/retro_contest_agent/blob/master/fastlearner/ppo2ttifrutti_sonic_env.py
import numpy as np
import gym
# library used to modify frames
import cv2
# pywinauto handles windows application and 'builds' game environment
# requires python 3.7.4
from pywinauto.application import Application
# for capturing images quickly
import win32gui, win32ui, win32con, win32api
#sending inputs
import ctypes
# used to set sleep timers between actions (to not overwrite each other)
import time

# fix later
import warnings
warnings.filterwarnings("ignore")

# source to this solution and code:
# https://pythonprogramming.net/direct-input-game-python-plays-gta-v/
# http://www.gamespp.com/directx/directInputKeyboardScanCodes.html
SendInput = ctypes.windll.user32.SendInput
PUL = ctypes.POINTER(ctypes.c_ulong)
class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]

class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time",ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                 ("mi", MouseInput),
                 ("hi", HardwareInput)]

class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]


# class to handle stardew valley environment // gym class // works by itself
class svEnv(gym.Env):
    # input hexcodes 
    C = 0x2E
    # [Y-top:Y-bottom, X-left:Xright] || reward region
    # 100 % zoom
    yo = 130
    yu = 788
    xl = 773
    xr = 813

    def __init__(self,
                 client = "Stardew Valley",
                 seconds_per_episode = 20,
                 show_preview = True):

        self.client = client
        self.seconds_per_episode = seconds_per_episode
        # variables for other implementations from the web in order to avoid errors
        # gym compliance
        self.reward_range = (0,10)
        self.metadata = None
        self.info = []
        # reward extraction defaults
        self.reward = 0
        self.reward_tracker = 0
        # done flag
        self.done = False
        # debugging purposes // showing more cv2 windows
        self.show_preview = show_preview
        # resetting at the end of initialization in order to identify window
        self.reset()
    
    def close(self):
        '''
        Closing the game or minimizing the game could be implemented here.
        This method is here to avoid crashes when testing reinforcement learning implementations from the web (gym compliance).
        '''
        print('Closing game')
        cv2.destroyAllWindows()
        return
    
    def render(self, mode='human', close=False):
        '''
        The game will always be rendered, since it needs to be the top window to send inputs and extract rewards
        This method is here to avoid crashes when testing implementations from the web.
        '''
        print('Rendering game.')
        return
        
    def reset(self):
       '''
       Resetting the environment means bringing the agent to the same point from where an episode starts.
       Method is expected to return the initial observation by gym standards.
       '''
       try:
           self.app = Application().connect(title_re=self.client)
           self.app.top_window().set_focus()
           #self.app.window().move_window(x=-10, y=30)
       except:
           raise OSError('Could not find environment window. Make sure to open the environment first or close windows which share the same name.')            
       
       
       # grab img for initial observation
       img = self.grab_screen(region=(self.xl,self.yo,self.xr,self.yu))
       self.episode_start = time.time()
       return img

    def step(self, action):
        
        self.done = False
        done = False
        action = str(action) #probably dumbest way of doing this
        #print("action: {}".format(action))
        if action == '[ True False]':
            self.PressKey(self.C)
            print("use tool")
        if action == '[False  True]':
            self.ReleaseKey(self.C)
        
        
        
        # observe environment
        img =  self.grab_screen(region=(self.xl,self.yo,self.xr,self.yu))
        # reward calculation
        reward = self.reward + 1
        self.reward = reward
        
        # done flag in order to go to next episode
        # should be 'if tee == frozen: done = True' instead of being based on time
        if self.done == True:
            done = True
            reward = 0
            print('catching done')
        
        return img, reward, done, self.info

    # image grabbing by Frannecklp
    # https://pythonprogramming.net/next-steps-python-plays-gta-v/
    def grab_screen(self, region=None):
        hwin = win32gui.GetDesktopWindow()

        if region:
                left,top,x2,y2 = region
                width = x2 - left
                height = y2 - top
        else:
            width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
            height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
            left = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
            top = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)

        hwindc = win32gui.GetWindowDC(hwin)
        srcdc = win32ui.CreateDCFromHandle(hwindc)
        memdc = srcdc.CreateCompatibleDC()
        bmp = win32ui.CreateBitmap()
        bmp.CreateCompatibleBitmap(srcdc, width, height)
        memdc.SelectObject(bmp)
        memdc.BitBlt((0, 0), (width, height), srcdc, (left, top), win32con.SRCCOPY)

        signedIntsArray = bmp.GetBitmapBits(True)
        img = np.frombuffer(signedIntsArray, dtype='uint8')
        img = img.reshape((self.yu-self.yo,self.xr-self.xl,4))
        
        srcdc.DeleteDC()
        memdc.DeleteDC()
        win32gui.ReleaseDC(hwin, hwindc)
        win32gui.DeleteObject(bmp.GetHandle())
        
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

        #preview
        if self.show_preview == True:
            cv2.imshow('Processed_Video', img)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()

        img = img.flatten() 
        return img
    
    # source to this solution and code:
    # https://pythonprogramming.net/direct-input-game-python-plays-gta-v/
    def PressKey(self, hexKeyCode):
        extra = ctypes.c_ulong(0)
        ii_ = Input_I()
        ii_.ki = KeyBdInput( 0, hexKeyCode, 0x0008, 0, ctypes.pointer(extra) )
        x = Input( ctypes.c_ulong(1), ii_ )
        ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
    
    def ReleaseKey(self, hexKeyCode):
        extra = ctypes.c_ulong(0)
        ii_ = Input_I()
        ii_.ki = KeyBdInput( 0, hexKeyCode, 0x0008 | 0x0002, 0, ctypes.pointer(extra) )
        x = Input( ctypes.c_ulong(1), ii_ )
        ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

    '''
    # source + windows settings 
    # https://stackoverflow.com/questions/50601200/pyhon-directinput-mouse-relative-moving-act-not-as-expected
    def MouseMoveTo(self, x, y):
        extra = ctypes.c_ulong(0)
        ii_ = Input_I()
        ii_.mi = MouseInput(x, y, 0, 0x0001, 0, ctypes.pointer(extra))
        command = Input(ctypes.c_ulong(0), ii_)
        ctypes.windll.user32.SendInput(1, ctypes.pointer(command), ctypes.sizeof(command))
    '''
    
    


if __name__ == '__main__':
    print('Executing this file should put Stardew Valley to the foreground and move the window to the top left corner.')
    print('This is will cause more windows to be opened for more insights.')
    print('You can take some snapshots of the windows by pressing q when selecting a cv2 img')
    env = svEnv(show_preview = True)
    for _ in range(5):
        env.step('nothing')
        cv2.waitKey(0) & 0xFF == ord('q')
        #env.hook_up()
    cv2.destroyAllWindows()

    