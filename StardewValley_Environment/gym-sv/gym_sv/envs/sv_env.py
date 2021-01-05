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
import win32gui
import win32ui
import win32con
import win32api
#sending inputs
import ctypes
# used to set sleep timers between actions (to not overwrite each other)
import time
# for left clicking
import pynput

# fix later
import warnings
warnings.filterwarnings("ignore")

import sys


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
    E = 0x12
    W = 0x11
    A = 0x1E
    S = 0x1F
    D = 0x20
    X = 0x2D
    # fishing mini game region [Y-top:Y-bottom, X-left:Xright]
    # 100 % zoom
    yo = 200
    yu = 770
    xl = 800
    xr = 825

    # exclamation mark region [Y-top:Y-bottom, X-left:Xright]
    # 100 % zoom
    eyo = 392
    eyu = 436
    exl = 957
    exr = 977

    # conditions for resetting day
    imgequalflag = 1.0
    imgtiredflag = 0.1

    # reporting
    hooked = 0
    nofish = 0

    # template to detect, if we hooked a fish
    exclamation = cv2.imread('exclamation.png',cv2.IMREAD_GRAYSCALE)
    fish = cv2.imread('fish.png',cv2.IMREAD_GRAYSCALE)
    donefish = cv2.imread('donefish.png',cv2.IMREAD_GRAYSCALE)
    imgequal = cv2.imread('imgequal.png',cv2.IMREAD_GRAYSCALE)
    tired = cv2.imread('tired.png',cv2.IMREAD_GRAYSCALE)
    blobber = cv2.imread('blobber.png',cv2.IMREAD_GRAYSCALE)
    twelve = cv2.imread('twelve.png',cv2.IMREAD_GRAYSCALE)
    am = cv2.imread('am.png',cv2.IMREAD_GRAYSCALE)
    obs = cv2.imread('obs.png',cv2.IMREAD_GRAYSCALE)


    def __init__(self,
                 client="Stardew Valley"):

        self.client = client
        # variables for other implementations from the web in order to avoid errors
        # gym compliance
        self.reward_range = (0, 10)
        self.metadata = None
        self.info = []
        # reward extraction defaults
        self.reward = 1
        # done flag

        self.done = True
        self.stepcounter = -1

    def close(self):
        '''
        Closing the game or minimizing the game could be implemented here.
        This method is here to avoid crashes when testing reinforcement learning implementations 
        from the web (gym compliance).
        '''
        print('Closing game')
        cv2.destroyAllWindows()
        return
    
    def render(self, mode='human', close=False):
        '''
        The game will always be rendered, since it needs to be the top window 
        to send inputs and extract rewards
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
       except:
           raise OSError('Could not find environment window. Make sure to open the environment first or close windows which share the same name.')            
       
       
       # get into starting position
       self.resetday()

       # basic img to avoid errors.
       # This is the same observation that would be made anyways
       # self.obs will be overwritten in prgramm flow
       img = self.obs
       return img

    def step(self, action):
        #print("enter step function")

        if self.done == False:
            # these are real steps

            stepcounter += 1

            #print("in step function, catching fish")

            # action decoding
            if action[0] == True:
                self.PressKey(self.C)
                #print("step action: press c")
            if action[1] == True:
                self.ReleaseKey(self.C)
                #print("step action: release c")
            
            # observation
            img = self.grab_fishing_screen()
            self.obs = img
            reward = self.reward

            # reward calculation
            reward = self.reward
            if reward == 0:
                reward = -1
            print("reward: ", reward)

            if stepcounter > 1500:
                time.sleep(0.5)
                # close fishing mini game
                self.PressAndReleaseKey(self.E)
                time.sleep(0.5)
                # ends episode
                #change naming convention
                done = true

        else:
            # we expect to enter this else ones after catching or losing a fish
            # input (action) doesn't matter
            # we give the last observation and reward
            # this kind of makes sense with the fishing game and the momentum
            # of the green bar in mind
            img = self.obs

            reward = self.reward
            if reward == 0:
                reward = -1
            print("step: {}/1500, reward: {:.2f}".format(stepcounter, reward))

            # click away message, if we caught a fish
            # don't click anything, if we didn't catch a fish
            time.sleep(0.5)
            self.PressAndReleaseKey(self.X)
            time.sleep(0.5)


            # still need to hook a fish
            self.hookfish()







        # buggy as hell, logic error
        '''
        # done flag in order to go to next episode
        if self.done == True:
            # done flag increases episode counter, but also resets environment
            #done = True
            reward = 0
            self.ReleaseKey(self.C)
            print('catching done')
            time.sleep(2)
            self.PressAndReleaseKey(self.X)
            time.sleep(1)
            self.hookfish()
        '''

        done = False

        return img, reward, done, self.info

    def resetday(self):

        print("Things hooked today: {}".format(self.hooked))
        print("Fish hooked today: {}".format(self.hooked - self.nofish))
        self.hooked = 0
        self.nofish = 0

        time.sleep(1)
        # open inventory
        self.PressAndReleaseKey(self.E)
        #reset mouse to top left of left monitor 
        self.MouseMoveTo(-9999, 0)
        self.MouseMoveTo(0, -9999)
        self.MouseMoveTo(-9999, -9999)
        # move mouse to monitor 1
        self.MouseMoveTo(1280, 0)
        # move mouse to Exit Game
        self.MouseMoveTo(1064, 250)
        time.sleep(0.5)
        self.left_click()
        # move mouse to Exit To Title
        self.MouseMoveTo(-150, 220)
        time.sleep(0.5)
        self.left_click()
        # move mouse to Load
        self.MouseMoveTo(-80, 470)
        time.sleep(0.5)
        self.left_click()
        # move mouse to Select Playthrough 1
        self.MouseMoveTo(0, -650)
        time.sleep(0.5)
        self.left_click()
        time.sleep(3)

        # walk left 
        self.PressKey(self.A)
        time.sleep(1.2)
        self.ReleaseKey(self.A)
        time.sleep(0.5)
        # walk down 
        self.PressKey(self.S)
        time.sleep(8.5)
        self.ReleaseKey(self.S)
        time.sleep(0.5)
        # walk right 
        self.PressKey(self.D)
        time.sleep(2.9)
        self.ReleaseKey(self.D)
        time.sleep(0.5)

    def hookfish(self):

        # check if inventory is full
            # if yes, reset day
        # check if it is late
            # if yes, reset day
        
        # cast fishing rod
            # tired?
                # if yes, reset day
            # bobber in water?
                # if yes, ok go ahead
                # if no, recast fishing rod
        
        # wait for fish
        # if exclamation mark
            #start reeling in fish
        
        # check if we hooked a fish
            # if not: press c
            # of yes:
                # start step() until no fish on screen

        resetcondition = self.check_for_reset()

        if not resetcondition:
        
            print("casting rod now")
            polecasted = False
            while polecasted == False:
                self.PressKey(self.C)
                time.sleep(1.04)
                self.ReleaseKey(self.C)
                time.sleep(1)
                polecasted = self.grab_blobber_screen()
        
            print("waiting for fish")
            hook = False
            while hook == False:
                hook = self.grab_exclamation_screen()
        
            # reel in rod
            self.PressAndReleaseKey(self.C)

            # sleep a little bit to account for animation
            time.sleep(1.5)

            # check if we hooked a fish
            # grab fishing screen in order to update reward

            while self.done == True:
                #print("self.done: ", self.done)
                img = self.grab_fishing_screen()

                if self.done == True:
                    print("but it wasn't a fish")
                    self.nofish += 1
                    
                    time.sleep(0.5)

                    # click away message
                    self.PressAndReleaseKey(self.C)

                    time.sleep(0.5)

                    print("casting rod now")

                    polecasted = False
                    while polecasted == False:
                        self.PressKey(self.C)
                        time.sleep(1.04)
                        self.ReleaseKey(self.C)
                        time.sleep(1)
                        polecasted = self.grab_blobber_screen()
                
                    print("waiting for fish")
                    hook = False
                    while hook == False:
                        hook = self.grab_exclamation_screen()
                
                    # reel in rod
                    self.PressAndReleaseKey(self.C)

                    # sleep a little bit to account for animation
                    time.sleep(1.5)

            print("we hooked a fish, let's catch it.")
            #return img
        
        if resetcondition:
            self.resetday()

    def grab_fishing_screen(self):
        hwin = win32gui.GetDesktopWindow()

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
        
        # 4224 is total pixel width
        # 1080 is total monitor height
        img = img.reshape((1080,4224,4))
        
        srcdc.DeleteDC()
        memdc.DeleteDC()
        win32gui.ReleaseDC(hwin, hwindc)
        win32gui.DeleteObject(bmp.GetHandle())
        
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

        # 1280 accounts for left monitor
        img = img[self.yo:self.yu, 1280+self.xl:1280+self.xr]

        # done fishing
        result=cv2.matchTemplate(img,self.donefish,cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        #print("done, when not 1.0: ",max_val)
        if max_val > 0.9:
            #print("fish on screen max_val:",max_val)
            # fish is on screen
            self.done = False

            # reward
            _ , cpimg = cv2.threshold(img, 170, 240, cv2.THRESH_BINARY)
            result=cv2.matchTemplate(cpimg,self.fish,cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            
            #print(max_val)
            if max_val > 0.52:
                self.reward = max_val
            else:
                self.reward = 0

            #cv2.imwrite("test.png", img)
            #sys.exit()

        else:
            self.done = True

        img = img.flatten() 
        return img

    def grab_exclamation_screen(self):
        hwin = win32gui.GetDesktopWindow()

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

        # 4224 is total pixel width
        # 1080 is total monitor height
        img = img.reshape((1080,4224,4))
        
        srcdc.DeleteDC()
        memdc.DeleteDC()
        win32gui.ReleaseDC(hwin, hwindc)
        win32gui.DeleteObject(bmp.GetHandle())
        
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

        # 1280 accounts for left monitor
        img = img[self.eyo:self.eyu, 1280+self.exl:1280+self.exr]

        # find exclamation marc
        result=cv2.matchTemplate(img,self.exclamation,cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        #print(max_val)
        hook = False
        if max_val > 0.50:
            hook = True
            print("hooked something")
            self.hooked += 1

        return hook

    def grab_blobber_screen(self):
        hwin = win32gui.GetDesktopWindow()

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

        # 4224 is total pixel width
        # 1080 is total monitor height
        img = img.reshape((1080,4224,4))
        
        srcdc.DeleteDC()
        memdc.DeleteDC()
        win32gui.ReleaseDC(hwin, hwindc)
        win32gui.DeleteObject(bmp.GetHandle())
        
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

        # blobber area
        img = img[510:560, 1280+1050:1280+1250]
    
        # find exclamation marc
        result=cv2.matchTemplate(img,self.blobber,cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        blobber = False

        if max_val > 0.80:
            blobber = True
        else:
            print("blobber not in water")
            self.resetday()

        #img = img.flatten() 
        return blobber

    def check_for_reset(self):

        hwin = win32gui.GetDesktopWindow()

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

        # 4224 is total pixel width
        # 1080 is total monitor height
        img = img.reshape((1080,4224,4))
        
        srcdc.DeleteDC()
        memdc.DeleteDC()
        win32gui.ReleaseDC(hwin, hwindc)
        win32gui.DeleteObject(bmp.GetHandle())
        
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

        resetcondition = False
        # ------------
        imgequal = img[956:1016, 1280+1220:1280+1280]
        result=cv2.matchTemplate(imgequal,self.imgequal,cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)


        if max_val < 0.6:
            print("inventory full, resetting day")
            #print("imgequal less than 0.6: {}".format(max_val))
            resetcondition = True

        if not resetcondition:
            # ------------
            tired = img[995:1005, 1280+1880:1280+1895]
            #cv2.imwrite("test.png", tired)
            result=cv2.matchTemplate(tired,self.tired,cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            if max_val > 0.99:
                print(r"I'm tired, resetting day")
                #print("imgtired over 0.4: {}".format(max_val))
                resetcondition = True

        if not resetcondition:
            # ------------
            # 12 : xx am coords
            late = img[147:179, 1280+1731:1280+1882]
            # check if 12
            result=cv2.matchTemplate(late,self.twelve,cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            if max_val > 0.95:
                result=cv2.matchTemplate(late,self.am,cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(result)
                if max_val > 0.95:
                    print(r"It's late, resetting day")
                    resetcondition = True
       
        #cv2.imwrite("test.png", late)
        #sys.exit()
        return resetcondition

    def PressAndReleaseKey(self, hexKeyCode):
        self.PressKey(hexKeyCode)
        time.sleep(0.2)
        self.ReleaseKey(hexKeyCode)
        time.sleep(0.1)
    
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

    # from: https://www.reddit.com/r/learnpython/comments/bognbs/direct_input_for_python_with_pynput/
    def left_click(self):
        time.sleep(1)

        extra = ctypes.c_ulong(0)
        ii_ = pynput._util.win32.INPUT_union()
        ii_.mi = pynput._util.win32.MOUSEINPUT(0, 0, 0, 0x0002, 0, ctypes.cast(ctypes.pointer(extra), ctypes.c_void_p))
        x=pynput._util.win32.INPUT(ctypes.c_ulong(0), ii_)
        SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
        
        time.sleep(1)

        extra = ctypes.c_ulong(0)
        ii_ = pynput._util.win32.INPUT_union()
        ii_.mi = pynput._util.win32.MOUSEINPUT(0, 0, 0, 0x0004, 0, ctypes.cast(ctypes.pointer(extra), ctypes.c_void_p))
        x=pynput._util.win32.INPUT(ctypes.c_ulong(0), ii_)
        SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

        time.sleep(1)

    # source + windows settings 
    # https://stackoverflow.com/questions/50601200/pyhon-directinput-mouse-relative-moving-act-not-as-expected
    def MouseMoveTo(self, x, y):
        extra = ctypes.c_ulong(0)
        ii_ = Input_I()
        ii_.mi = MouseInput(x, y, 0, 0x0001, 0, ctypes.pointer(extra))
        command = Input(ctypes.c_ulong(0), ii_)
        ctypes.windll.user32.SendInput(1, ctypes.pointer(command), ctypes.sizeof(command))

    
    


if __name__ == '__main__':
    print('Executing this file should put Stardew Valley to the foreground.')
    print('The player character must be in a non-busy state.')
    print('It should reset the day and hook a fish.')
    print('It will step without action for 512 steps to demonstrate the environment.')

    env = svEnv()
    env.reset()
    for _ in range(512):
        env.step('nothing')


    print("preview over")

    