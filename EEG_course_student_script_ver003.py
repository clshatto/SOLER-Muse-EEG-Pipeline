#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy2 Experiment Builder (v1.90.2),
    on September 08, 2021, at 14:34
If you publish work using this script please cite the PsychoPy publications:
    Peirce, JW (2007) PsychoPy - Psychophysics software in Python.
        Journal of Neuroscience Methods, 162(1-2), 8-13.
    Peirce, JW (2009) Generating stimuli for neuroscience using PsychoPy.
        Frontiers in Neuroinformatics, 2:10. doi: 10.3389/neuro.11.010.2008
"""

from __future__ import absolute_import, division

import psychopy
#psychopy.useVersion('1.90.2')

from psychopy import locale_setup, sound, gui, visual, core, data, event, logging, clock
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER)
import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle
import os  # handy system and path functions
import sys  # to get file system encoding

#starting hxf 9/8/2021
import time #for getting local computer time
synLocalTime='12:20:00' #i.e., '10:15:00', 15:30:30 will start fixation at 10:15AM make sure set all students' laptop local time correctly!
from pylsl import StreamInfo, StreamOutlet, local_clock
# create
info = StreamInfo('Markers', 'Markers', 1, 0, 'int32', 'myuidw43536')
# next make an outlet
outlet = StreamOutlet(info)
markernames = {'experiment_started':111,'fixation_start':201,'fixation_stopped':202,'responseA_start':211,'responseA_stopped':212,'responseB_start':221,'responseB_stopped':222,'responseC_start':231,'responseC_stopped':232,'responseD_start':241,'responseD_stopped':242,'responseIncorrect_start':251,'responseIncorrect_stopped':252,'experiment_stopped':999}
#				%% Marker List
#				% 111: experiment start marker in order to let lsl-record.py stop recording
#				% 201: Fixation onset (associated with stimulus)
#				% 202: fixation_stopped: Fixation offset
#				% 211: responseA_start (i.e., press 'a' for the questionaire)
#				% 212: responseA_stopped
#				% 221: responseB_start (i.e., press 'b' for the questionaire)
#				% 222: responseB_stopped
#				% 231: responseC_start (i.e., press 'c' for the questionaire)
#				% 232: responseC_stopped
#				% 241: responseD_start (i.e., press 'd' for the questionaire)
#				% 242: responseD_stopped
#				% 251: responseIncorrect_start, i.e., press any key but not allowed answer 'a' 'b' 'c' or 'd
#				% 252: responseIncorrect_stopped
#				% 999: experiment stop marker in order to let lsl-record.py stop recording
#trigger_status=0 #stopped, 1 means started, not used in this task
#note: responseA/B/C/D/Incorrect_stopped was not used
#       responseIncorrect_start is also not used since line#287 only accepts 'abcd' and '1' keys, i.e., theseKeys = event.getKeys(keyList=['a', 'b', 'c', 'd', '1'])
#ending by hxf 9/8/2021

# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)

# Store info about the experiment session
expName = 'student_script_ver003'  # from the Builder filename that created this script
expInfo = {'participant': '', 'session': '001'}
dlg = gui.DlgFromDict(dictionary=expInfo, title=expName)
if dlg.OK == False:
    core.quit()  # user pressed cancel
expInfo['date'] = data.getDateStr()  # add a simple timestamp
expInfo['expName'] = expName

# Data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
filename = _thisDir + os.sep + u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])

# An ExperimentHandler isn't essential but helps with data saving
thisExp = data.ExperimentHandler(name=expName, version='',
    extraInfo=expInfo, runtimeInfo=None,
    originPath=None,
    savePickle=True, saveWideText=True,
    dataFileName=filename)
# save a log file for detail verbose info
logFile = logging.LogFile(filename+'.log', level=logging.EXP)
logging.console.setLevel(logging.WARNING)  # this outputs to the screen, not a file

endExpNow = False  # flag for 'escape' or other condition => quit the exp

# Start Code - component code to be run before the window creation

# Setup the Window
win = visual.Window(
    size=[800, 450], fullscr=False, screen=0,
    allowGUI=False, allowStencil=False,
    monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
    blendMode='avg', useFBO=True,
    units='height')
# store frame rate of monitor if we can measure it
expInfo['frameRate'] = win.getActualFrameRate()
if expInfo['frameRate'] != None:
    frameDur = 1.0 / round(expInfo['frameRate'])
else:
    frameDur = 1.0 / 60.0  # could not measure, so guess

# Initialize components for Routine "Introduction"
IntroductionClock = core.Clock()
# text_introduction = visual.TextStim(win=win, name='text_introduction',
#     text='Good morning!\n\nPlease focus on the presenter’s screen and follow instructions.',
#     font='Arial',
#     pos=(0, 0), height=0.1, wrapWidth=None, ori=0,
#     color='white', colorSpace='rgb', opacity=1,
#     depth=0.0);

#hxf 9/8/2021
text_introduction = visual.TextStim(win=win, name='text_introduction',
    text="Good morning!\n\nPlease focus on the presenter's screen and follow instructions.",
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0,
    color='white', colorSpace='rgb', opacity=1,
    depth=0.0);

# Initialize components for Routine "trial"
trialClock = core.Clock()
text_fixation = visual.TextStim(win=win, name='text_fixation',
    text='+',
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1,
    depth=0.0);

# Initialize components for Routine "ThankYou"
ThankYouClock = core.Clock()
text_thankyou = visual.TextStim(win=win, name='text_thankyou',
    text='Thank you for your time! \n\nHave a great rest of your day!',
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1,
    depth=0.0);

# Create some handy timers
globalClock = core.Clock()  # to track the time since experiment started
routineTimer = core.CountdownTimer()  # to track time remaining of each (non-slip) routine 

# ------Prepare to start Routine "Introduction"-------
t = 0
IntroductionClock.reset()  # clock
frameN = -1
continueRoutine = True
# update component parameters for each repeat
key_resp_trigger = event.BuilderKeyResponse()
# keep track of which components have finished
IntroductionComponents = [text_introduction, key_resp_trigger]
for thisComponent in IntroductionComponents:
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED

# -------Start Routine "Introduction"-------
while continueRoutine:
    # get current time
    t = IntroductionClock.getTime()
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *text_introduction* updates
    if t >= 0.0 and text_introduction.status == NOT_STARTED:
        # keep track of start time/frame for later
        text_introduction.tStart = t
        text_introduction.frameNStart = frameN  # exact frame index
        text_introduction.setAutoDraw(True)
    
    # *key_resp_trigger* updates
    if t >= 0.0 and key_resp_trigger.status == NOT_STARTED:
        # keep track of start time/frame for later
        key_resp_trigger.tStart = t
        key_resp_trigger.frameNStart = frameN  # exact frame index
        key_resp_trigger.status = STARTED
        # keyboard checking is just starting
        win.callOnFlip(key_resp_trigger.clock.reset)  # t=0 on next screen flip
        event.clearEvents(eventType='keyboard')
    if key_resp_trigger.status == STARTED:
        theseKeys = event.getKeys(keyList=['0'])
        
        # check for quit:
        if "escape" in theseKeys:
            endExpNow = True
        # #method1: triggered by press '0' manually
        # if len(theseKeys) > 0:  # at least one key was pressed
        #     key_resp_trigger.keys = theseKeys[-1]  # just the last key pressed
        #     key_resp_trigger.rt = key_resp_trigger.clock.getTime()
        #     # a response ends the routine
        #     continueRoutine = False
        #
        #     # send the started trigger, adding by hxf 10/12/2021
        #     timestamp = local_clock()  # don't use t
        #     outlet.push_sample([markernames['experiment_started']], timestamp)
        #     # ending by hxf 10/12/2021
        #method2: triggered by local computer time, hxf 10/12/2021
        # send the started trigger, adding by hxf 10/12/2021
        t_localTime = time.localtime()
        current_time = time.strftime("%H:%M:%S", t_localTime)
        #print(current_time) #e.g., 14:28:54
        if current_time == synLocalTime: #e.g., synLocalTime='10:15:00'
            key_resp_trigger.keys = 'auto' #i.e., automatically triggered
            key_resp_trigger.rt = key_resp_trigger.clock.getTime()
            # a response ends the routine
            continueRoutine = False

            # send the started trigger, adding by hxf 10/12/2021
            timestamp = local_clock()  # don't use t
            outlet.push_sample([markernames['experiment_started']], timestamp)
            # ending by hxf 10/12/2021
        #ending by hxf 10/12/2021

    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in IntroductionComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # check for quit (the Esc key)
    if endExpNow or event.getKeys(keyList=["escape"]):
        # send the stopped trigger, adding by hxf 11/19/2021
        timestamp = local_clock()  # don't use t, Obtain a local system time stamp in seconds.
        outlet.push_sample([markernames['experiment_stopped']], timestamp)
        # ending by hxf 11/19/2021
        core.quit()
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "Introduction"-------
for thisComponent in IntroductionComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
# check responses
if key_resp_trigger.keys in ['', [], None]:  # No response was made
    key_resp_trigger.keys=None
thisExp.addData('key_resp_trigger.keys',key_resp_trigger.keys)
if key_resp_trigger.keys != None:  # we had a response
    thisExp.addData('key_resp_trigger.rt', key_resp_trigger.rt)
thisExp.nextEntry()
# the Routine "Introduction" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# set up handler to look after randomisation of conditions etc
recordingloop = data.TrialHandler(nReps=1, method='random', 
    extraInfo=expInfo, originPath=-1,
    trialList=[None],
    seed=None, name='recordingloop')
thisExp.addLoop(recordingloop)  # add the loop to the experiment
thisRecordingloop = recordingloop.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb = thisRecordingloop.rgb)
if thisRecordingloop != None:
    for paramName in thisRecordingloop:
        exec('{} = thisRecordingloop[paramName]'.format(paramName))

for thisRecordingloop in recordingloop:
    currentLoop = recordingloop
    # abbreviate parameter names if possible (e.g. rgb = thisRecordingloop.rgb)
    if thisRecordingloop != None:
        for paramName in thisRecordingloop:
            exec('{} = thisRecordingloop[paramName]'.format(paramName))
    
    # ------Prepare to start Routine "trial"-------
    t = 0
    trialClock.reset()  # clock
    frameN = -1
    continueRoutine = True
    # update component parameters for each repeat
    key_resp_questionnaire = event.BuilderKeyResponse()
    # keep track of which components have finished
    trialComponents = [text_fixation, key_resp_questionnaire]
    for thisComponent in trialComponents:
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    
    # -------Start Routine "trial"-------
    #hxf 9/8/2021
    responseKeys=[]
    responseTime=[]
    while continueRoutine:
        # get current time
        t = trialClock.getTime()
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_fixation* updates
        if t >= 0.0 and text_fixation.status == NOT_STARTED:
            # keep track of start time/frame for later
            text_fixation.tStart = t
            text_fixation.frameNStart = frameN  # exact frame index
            text_fixation.setAutoDraw(True)

            # adding by hxf 10/2121/2021
            # for fixation onset
            # sending starting fixation marker
            timestamp = local_clock()  # don't use t
            outlet.push_sample([markernames['fixation_start']], timestamp)
            # ending by hxf 10/12/2021
        
        # *key_resp_questionnaire* updates
        if t >= 0.0 and key_resp_questionnaire.status == NOT_STARTED:
            # keep track of start time/frame for later
            key_resp_questionnaire.tStart = t
            key_resp_questionnaire.frameNStart = frameN  # exact frame index
            key_resp_questionnaire.status = STARTED
            # keyboard checking is just starting
            win.callOnFlip(key_resp_questionnaire.clock.reset)  # t=0 on next screen flip
            event.clearEvents(eventType='keyboard')
        if key_resp_questionnaire.status == STARTED:
            theseKeys = event.getKeys(keyList=['a', 'b', 'c', 'd', '1'])
            
            # check for quit:
            if "escape" in theseKeys:
                endExpNow = True

            if len(theseKeys) > 0:  # at least one key was pressed
                key_resp_questionnaire.keys = theseKeys[-1]  # just the last key pressed
                key_resp_questionnaire.rt = key_resp_questionnaire.clock.getTime()
                # a response ends the routine
                #by hxf 9/8/2021, so it's a endless trial, until you press key '1' to finish
                # continueRoutine = False #will exit this routine
                if key_resp_questionnaire.keys=='1': #then exist this routine
                    continueRoutine = False

                    # send the fixation stop trigger, adding by hxf 10/12/2021
                    timestamp = local_clock()  # don't use t
                    outlet.push_sample([markernames['fixation_stopped']], timestamp)
                    # ending by hxf 10/12/2021

                else: #save the response to the output .csv file
                    # #method1: still can't record the response key to the output .csv file since recordingloop.addData only save one time response data
                    # key_resp_questionnaire.keys = theseKeys #use all keys, [-1] is just the last key pressed
                    # key_resp_questionnaire.rt = key_resp_questionnaire.clock.getTime()
                    # # check responses
                    # if key_resp_questionnaire.keys in ['', [], None]:  # No response was made
                    #     key_resp_questionnaire.keys = None
                    # recordingloop.addData('key_resp_questionnaire.keys', key_resp_questionnaire.keys)
                    # if key_resp_questionnaire.keys != None:  # we had a response
                    #     recordingloop.addData('key_resp_questionnaire.rt', key_resp_questionnaire.rt)
                    # print("theseKeys:")
                    # print(theseKeys)  # ['a']
                    # print("theseKeys[-1]:")
                    # print(theseKeys[-1])  # a
                    # method2: define responseKeys and responseTime lists and save them later outside this routine
                    #responseKeys.append(key_resp_questionnaire.keys)  # just the last key pressed
                    responseKeys.append(theseKeys[-1])  # just the last key pressed
                    responseTime.append(key_resp_questionnaire.rt)  # add response time
                    # print("responseKeys:")
                    # print(responseKeys)
                    # print("responseTime:")
                    # print(responseTime)

                    # send the response trigger, adding by hxf 10/12/2021
                    timestamp = local_clock()  # don't use t
                    if (key_resp_questionnaire.keys == 'a' or key_resp_questionnaire.keys == 'A'):
                        outlet.push_sample([markernames['responseA_start']], timestamp)
                    elif (key_resp_questionnaire.keys == 'b' or key_resp_questionnaire.keys == 'B'):
                        outlet.push_sample([markernames['responseB_start']], timestamp)
                    elif (key_resp_questionnaire.keys == 'c' or key_resp_questionnaire.keys == 'C'):
                        outlet.push_sample([markernames['responseC_start']], timestamp)
                    elif (key_resp_questionnaire.keys == 'd' or key_resp_questionnaire.keys == 'D'):
                        outlet.push_sample([markernames['responseD_start']], timestamp)
                    else: #non abcd or ABCD response, actually not used since line#287 only accepts 'abcd' and '1' keys, i.e., theseKeys = event.getKeys(keyList=['a', 'b', 'c', 'd', '1'])
                        outlet.push_sample([markernames['responseIncorrect_start']], timestamp)
                    # ending by hxf 10/12/2021

        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in trialComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # check for quit (the Esc key)
        if endExpNow or event.getKeys(keyList=["escape"]):
            # send the stopped trigger, adding by hxf 11/19/2021
            timestamp = local_clock()  # don't use t, Obtain a local system time stamp in seconds.
            outlet.push_sample([markernames['experiment_stopped']], timestamp)
            # ending by hxf 11/19/2021
            core.quit()
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "trial"-------
    for thisComponent in trialComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # check responses
    # #method1: default, save the latest response key and it's response time (i.e., when exit the routine)
    # if key_resp_questionnaire.keys in ['', [], None]:  # No response was made
    #     key_resp_questionnaire.keys=None
    # recordingloop.addData('key_resp_questionnaire.keys',key_resp_questionnaire.keys)
    # if key_resp_questionnaire.keys != None:  # we had a response
    #     recordingloop.addData('key_resp_questionnaire.rt', key_resp_questionnaire.rt)
    #method2: save all keys and their response time, hxf 9/8/2021
    if responseKeys in ['', [], None]:  # No response was made
        responseKeys=None
    recordingloop.addData('key_resp_questionnaire.keys',responseKeys)
    if responseKeys != None:  # we had a response
        recordingloop.addData('key_resp_questionnaire.rt', responseTime)

    # the Routine "trial" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    thisExp.nextEntry()
    
# completed 1 repeats of 'recordingloop'


# ------Prepare to start Routine "ThankYou"-------
t = 0
ThankYouClock.reset()  # clock
frameN = -1
continueRoutine = True
routineTimer.add(5.000000)
# update component parameters for each repeat
# keep track of which components have finished
ThankYouComponents = [text_thankyou]
for thisComponent in ThankYouComponents:
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED

# -------Start Routine "ThankYou"-------
while continueRoutine and routineTimer.getTime() > 0:
    # get current time
    t = ThankYouClock.getTime()
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *text_thankyou* updates
    if t >= 0.0 and text_thankyou.status == NOT_STARTED:
        # keep track of start time/frame for later
        text_thankyou.tStart = t
        text_thankyou.frameNStart = frameN  # exact frame index
        text_thankyou.setAutoDraw(True)
    frameRemains = 0.0 + 5- win.monitorFramePeriod * 0.75  # most of one frame period left
    if text_thankyou.status == STARTED and t >= frameRemains:
        text_thankyou.setAutoDraw(False)
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in ThankYouComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # check for quit (the Esc key)
    if endExpNow or event.getKeys(keyList=["escape"]):
        # send the stopped trigger, adding by hxf 11/19/2021
        timestamp = local_clock()  # don't use t, Obtain a local system time stamp in seconds.
        outlet.push_sample([markernames['experiment_stopped']], timestamp)
        # ending by hxf 11/19/2021
        core.quit()
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "ThankYou"-------
for thisComponent in ThankYouComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)


# send the stopped trigger, adding by hxf 10/12/2021
timestamp = local_clock()  # don't use t, Obtain a local system time stamp in seconds.
outlet.push_sample([markernames['experiment_stopped']], timestamp)
# ending by hxf 10/12/2021

# these shouldn't be strictly necessary (should auto-save)
thisExp.saveAsWideText(filename+'.csv')
thisExp.saveAsPickle(filename)
logging.flush()
# make sure everything is closed down
thisExp.abort()  # or data files will save again on exit
win.close()
core.quit()
