from google.oauth2.service_account import Credentials
import gspread
import pandas as pd
import tkinter as tk
from tkinter import ttk
import threading
import random
import numpy as np
from PIL import Image, ImageTk
import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os
from functools import partial
from pathlib import Path
from collections import OrderedDict
from timm.models import create_model
import utils
import modeling_finetune
import cv2
import mediapipe as mp
import video_transforms as video_transforms
import volume_transforms as volume_transforms
import base64
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime





def get_args():
    parser = argparse.ArgumentParser('VideoMAE fine-tuning and evaluation script for video classification', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_freq', default=100, type=int)

    # Model parameters
    parser.add_argument('--model', default='vit_base_dim512_no_depth_patch16_160', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--tubelet_size', type=int, default= 2)
    parser.add_argument('--input_size', default=160, type=int,
                        help='videos input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--attn_drop_rate', type=float, default=0.0, metavar='PCT',
                        help='Attention dropout rate (default: 0.)')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--disable_eval_during_finetuning', action='store_true', default=False)
    parser.add_argument('--model_ema', action='store_true', default=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', action='store_true', default=False, help='')

    parser.add_argument('--attn_type', default='local_global',
                        type=str, help='attention type for spatiotemporal modeling')
    parser.add_argument('--lg_region_size', type=int, nargs='+', default=(2,5,10),
                        help='region size (t,h,w) for local_global attention')
    parser.add_argument('--lg_first_attn_type', type=str, default='self', choices=['cross', 'self'],
                        help='the first attention layer type for local_global attention')
    parser.add_argument('--lg_third_attn_type', type=str, default='cross', choices=['cross', 'self'],
                        help='the third attention layer type for local_global attention')
    parser.add_argument('--lg_attn_param_sharing_first_third', action='store_true',
                        help='share parameters of the first and the third attention layers for local_global attention')
    parser.add_argument('--lg_attn_param_sharing_all', action='store_true',
                        help='share all the parameters of three attention layers for local_global attention')
    parser.add_argument('--lg_classify_token_type', type=str, default='region', choices=['org', 'region', 'all'],
                        help='the token type in final classification for local_global attention')
    parser.add_argument('--lg_no_second', action='store_true',
                        help='no second (inter-region) attention for local_global attention')
    parser.add_argument('--lg_no_third', action='store_true',
                        help='no third (local-global interaction) attention for local_global attention')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")

    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--layer_decay', type=float, default=0.75)

    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--num_sample', type=int, default=2,
                        help='Repeated_aug (default: 2)')
    parser.add_argument('--aa', type=str, default='rand-m7-n4-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m7-n4-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    # Evaluation parameters
    parser.add_argument('--crop_pct', type=float, default=None)
    parser.add_argument('--short_side_size', type=int, default=160)
    parser.add_argument('--test_num_segment', type=int, default=5)
    parser.add_argument('--test_num_crop', type=int, default=3)
    
    # Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--model_key', default='model|module', type=str)
    parser.add_argument('--model_prefix', default='', type=str)
    parser.add_argument('--init_scale', default=0.001, type=float)
    parser.add_argument('--use_mean_pooling', action='store_true')
    parser.set_defaults(use_mean_pooling=True)
    parser.add_argument('--use_cls', action='store_false', dest='use_mean_pooling')

    # Dataset parameters
    parser.add_argument('--data_path', default='/path/to/list_kinetics-400', type=str,
                        help='dataset path')
    parser.add_argument('--eval_data_path', default=None, type=str,
                        help='dataset path for evaluation')
    parser.add_argument('--nb_classes', default=7, type=int,
                        help='number of the classification types')
    parser.add_argument('--imagenet_default_mean_and_std', default=True, action='store_true')
    parser.add_argument('--num_segments', type=int, default= 1)
    parser.add_argument('--num_frames', type=int, default= 16)
    parser.add_argument('--sampling_rate', type=int, default= 4)
    parser.add_argument('--data_set', default='FERV39k', choices=['Kinetics-400', 'SSV2', 'UCF101', 'HMDB51','image_folder',
                        'DFEW', 'FERV39k', 'MAFW', 'RAVDESS', 'CREMA-D', 'ENTERFACE'],
                        type=str, help='dataset')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='model\checkpoint-best.pth',
                        help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--save_ckpt', action='store_true')
    parser.add_argument('--no_save_ckpt', action='store_false', dest='save_ckpt')
    parser.set_defaults(save_ckpt=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    parser.add_argument('--enable_deepspeed', action='store_true', default=False)


    parser.add_argument('--val_metric', type=str, default='acc1', choices=['acc1', 'acc5', 'war', 'uar', 'weighted_f1', 'micro_f1', 'macro_f1'],
                        help='validation metric for saving best ckpt')
    parser.add_argument('--depth', default=16, type=int,
                        help='specify model depth, NOTE: only works when no_depth model is used!')

    parser.add_argument('--save_feature', action='store_true', default=False)

    return parser.parse_args()



args = get_args()

window = tk.Tk()
window.title('GUI')
window.geometry('380x400+600+200')
window.resizable(False, False)

findFace = mp.solutions.face_detection.FaceDetection()




scope = ['https://www.googleapis.com/auth/spreadsheets']
creds = Credentials.from_service_account_file("key.json", scopes=scope)
gs = gspread.authorize(creds)
 
sheet = gs.open_by_url('https://docs.google.com/spreadsheets/d/1MT422mFKqI9Y03oTEwdTQol_mBWj47wk21cLQW1V7B4/edit#gid=0')
worksheet = sheet.get_worksheet(0)
user_n=0



def del_c(num1,num2) :
    worksheet.update(chr(67+num1*3)+str(num2+1), "")
    us_num=worksheet.acell(chr(66+num1*3)+"1").value
    worksheet.update(chr(66+num1*3)+"1",str(int(us_num)-1))
    os._exit(0)
def back(num1,num2,name):
    worksheet.update(chr(67+num1*3)+str(num2+1), "")
    us_num=worksheet.acell(chr(66+num1*3)+"1").value
    worksheet.update(chr(66+num1*3)+"1",str(int(us_num)-1))
    ChooseServer(1,name)
def faceBox(frame):#face bounding box
    frameRGB = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    height = frame.shape[0]
    width = frame.shape[1]
    results = findFace.process(frameRGB)
    myFaces = []
    if results.detections != None:
        for face in results.detections:
            bBox = face.location_data.relative_bounding_box
            x,y,w,h = int(bBox.xmin*width),int(bBox.ymin*height),int(bBox.width*width),int(bBox.height*height)
            myFaces.append((x,y,w,h))
    return myFaces
def client(name,ser_num):
    allus=worksheet.get_values(chr(66+ser_num*3)+':'+chr(67+ser_num*3))
    for i in range(0,10):
       if i<int(allus[0][0]):
           if allus[i][1]=="":
               user_n=i
               break
       else:
           user_n=i
           break
    worksheet.update(chr(67+ser_num*3)+str(user_n+1), name)
    worksheet.update(chr(68+ser_num*3)+str(user_n+1), "mood: normal")
    worksheet.update(chr(66+ser_num*3)+'1', str(int(allus[0][0])+1))
    win_c = tk.Tk()
    win_c.title(name)
    win_c.geometry('810x512+100+50')
    win_c.configure(bg='black')
    win_c.resizable(False, False)
    img1 = Image.open('./happy.png')
    tk_img1 = ImageTk.PhotoImage(img1)
    img2 = Image.open('./sad.png')
    tk_img2 = ImageTk.PhotoImage(img2)
    img3 = Image.open('./angry.png')
    tk_img3 = ImageTk.PhotoImage(img3)
    img4 = Image.open('./normal.png')
    tk_img4 = ImageTk.PhotoImage(img4)
    img5 = Image.open('./surprise.png')
    tk_img5 = ImageTk.PhotoImage(img5)
    img6 = Image.open('./avator.png')
    tk_img6 = ImageTk.PhotoImage(img6)
    img7 = Image.open('./disgust.png')
    tk_img7 = ImageTk.PhotoImage(img7)
    img8 = Image.open('./fear.png')
    tk_img8 = ImageTk.PhotoImage(img8)

    def UI():
        global qt1
        global wt
        qt1=0
        wt=0
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FPS, 30)

        device = torch.device(args.device)

        Emotions = ['happiness','sadness','neutral','anger','surprise','disgust','fear']



        ## 創建模型(MAE-DFER)
        model = create_model(
            args.model,
            preturn_cfg=None,
            num_classes=args.nb_classes,
            all_frames=args.num_frames * args.num_segments,
            tubelet_size=args.tubelet_size,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            attn_drop_rate=args.attn_drop_rate,
            drop_block_rate=None,
            use_mean_pooling=args.use_mean_pooling,
            init_scale=args.init_scale,
            depth=args.depth,
            attn_type=args.attn_type,
            lg_region_size=args.lg_region_size, lg_first_attn_type=args.lg_first_attn_type,
            lg_third_attn_type=args.lg_third_attn_type,
            lg_attn_param_sharing_first_third=args.lg_attn_param_sharing_first_third,
            lg_attn_param_sharing_all=args.lg_attn_param_sharing_all,
            lg_classify_token_type=args.lg_classify_token_type,
            lg_no_second=args.lg_no_second, lg_no_third=args.lg_no_third
        )


        model.to(device)


        model_without_ddp = model

        ##print("Model = %s" % str(model_without_ddp))


        utils.auto_load_model_eval(
            args=args, model=model, model_without_ddp=model_without_ddp,
        )

        ## preprocess
        data_transform = video_transforms.Compose([
            video_transforms.Resize(size=(160, 160), interpolation='bilinear'),
            # me: old, may have bug (heigh != width)
            # video_transforms.Resize(self.short_side_size, interpolation='bilinear'),
            video_transforms.CenterCrop(size=(160, 160)),
            volume_transforms.ClipToTensor(),
            video_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        ])

        ####

        Faces = []
        prediction = 'neutral'
        indices = [1, 3, 5, 7, 9, 11, 12, 14, 16, 18, 19, 21, 23, 25, 27, 29]



        mood=""
        user_i=user_n
        Ls=[]
        L1s=[]
        timer=0
        l_mood=-1
        alln=worksheet.col_values(3*(ser_num+1))
        us_ct=0
        for i in range(0,10):
            if us_ct<user_i and alln[i]!="":
                Lname=tk.Label(frames[i], text=alln[i],font=("Courier",12),bg='#dfffdb', height=2,justify=tk.CENTER)
                Lname.place(x=3, y=3)
                Ls.append(Lname)
                L1=tk.Label(frames[i],bg='#dfffdb', height=70)
                L1.place(x=65, y=18)
                us_ct+=1
                L1s.append(L1)
            else:
                frames[i].configure(bg='gray')
                Lname=tk.Label(frames[i], text="",font=("Courier",12),bg='gray', height=2,justify=tk.CENTER)
                Lname.place(x=3, y=3)
                Ls.append(Lname)
                L1=tk.Label(frames[i],bg='gray', height=80)
                L1.place(x=60, y=30)
                L1s.append(L1)
                L1s[i].config(image=tk_img6)
        Lau=tk.Label(frame1,text="觀眾: "+chr(48+user_i), bg='#1d551A',font=("隸書",20),fg='white')
        Lau.place(x=10,y=10)
        m1=tk.Label(frame1,image=tk_img1, bg='#1d551A',height=50)
        m1.place(x=10,y=100)
        m2=tk.Label(frame1,image=tk_img2, bg='#1d551A',height=50)
        m2.place(x=10,y=190)
        m3=tk.Label(frame1,image=tk_img3, bg='#1d551A',height=50)
        m3.place(x=10,y=280)
        m4=tk.Label(frame1,image=tk_img4, bg='#1d551A',height=50)
        m4.place(x=10,y=370)
        m5=tk.Label(frame1,image=tk_img5, bg='#1d551A',height=50)
        m5.place(x=210,y=100)
        m6=tk.Label(frame1,image=tk_img7, bg='#1d551A',height=50)
        m6.place(x=207,y=190)
        m7=tk.Label(frame1,image=tk_img8, bg='#1d551A',height=50)
        m7.place(x=205,y=280)
        ms=[]
        mc=[]
        for i in range(0,7):
                m=tk.Label(frame1,text=chr(48), bg='#1d551A',font=("隸書",20),fg='white')
                m.place(x=int(i/4)*200+120,y=90*((i%4)+1)+15)
                ms.append(m)
                mc.append(0)
                
        """
        while(cap.isOpened( )==0):
                print("未開鏡頭")
                L1s[user_i-1].config(text="未開鏡頭",font=("隸書",12),fg='black')
                cv2.waitKey(1000)
                cap = cv2.VideoCapture(0)
                cap.set(cv2.CAP_PROP_FPS, 30)
        """
        while True:
            if qt1==1:
                wt=1
                break
            
            _ , frame = cap.read()

            frameRGB = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            height = frame.shape[0]
            width = frame.shape[1]

            results = findFace.process(frameRGB)

 
            if results.detections != None:
                for face in results.detections:
                    bBox = face.location_data.relative_bounding_box
                    x,y,w,h = int(bBox.xmin*width),int(bBox.ymin*height),int(bBox.width*width),int(bBox.height*height)
                    ##cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                    #cv2.imshow('test',frame)
                    Faces.append((x,y,w,h))
        

            if(len(Faces)%30==0):
                
                images = list()

                for indice in indices:
                    Face = Faces[indice]

                    x,y,w,h =Face
                    ##cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                    ##cv2.imshow('test',frame)
            
                    faceExp = frame[y:y+h,x:x+w]

                    faceExp = cv2.cvtColor(faceExp, cv2.COLOR_BGR2RGB)
                    pil = Image.fromarray(faceExp)



                    images.append(pil)

                images = data_transform(images)

                images = images.reshape(-1,3,16,160,160) #[1, 3, 16, 160, 160]

                images = images.to(device, non_blocking=True)

                model.eval()


                with torch.no_grad():
                    output =  model(images)

                pred = output.argmax(1)
                prediction = Emotions[pred.item()]
                ##print(prediction)
                
                if timer>=2:
                    
                    if pred.item()==0 and l_mood!=0:
                        worksheet.update(chr(68+ser_num*3)+str(user_n+1), "mood: happy")
                        l_mood=0
                    elif pred.item()==1 and l_mood!=1:
                        worksheet.update(chr(68+ser_num*3)+str(user_n+1), "mood: sad")
                        l_mood=1
                    elif pred.item()==2 and l_mood!=2:
                        worksheet.update(chr(68+ser_num*3)+str(user_n+1), "mood: normal")
                        l_mood=2
                    elif pred.item()==3 and l_mood!=3:
                        worksheet.update(chr(68+ser_num*3)+str(user_n+1), "mood: angry")
                        l_mood=3
                    elif pred.item()==4 and l_mood!=4:
                        worksheet.update(chr(68+ser_num*3)+str(user_n+1), "mood: surprise")
                        l_mood=4
                    elif pred.item()==5 and l_mood!=5:
                        worksheet.update(chr(68+ser_num*3)+str(user_n+1), "mood: disgust")
                        l_mood=5
                    elif pred.item()==6 and l_mood!=6:
                        worksheet.update(chr(68+ser_num*3)+str(user_n+1), "mood: fear")
                        l_mood=6
                    """
                    x=random.randint(1,5)
                    if x==1:
                        worksheet.update(chr(68+ser_num*3)+str(user_n+1), "mood: happy")
                    elif x==2:
                        worksheet.update(chr(68+ser_num*3)+str(user_n+1), "mood: sad")
                    elif x==3:
                        worksheet.update(chr(68+ser_num*3)+str(user_n+1), "mood: angry")
                    elif x==4:
                        worksheet.update(chr(68+ser_num*3)+str(user_n+1), "mood: surprise")
                    elif x==5:
                        worksheet.update(chr(68+ser_num*3)+str(user_n+1), "mood: normal")
                    """
                    allm=worksheet.get_values(chr(66+ser_num*3)+':'+chr(68+ser_num*3))
                    if int(allm[0][0])>user_i:
                        L1s.clear()
                        Ls.clear()
                        frames.clear()
                        us_ct=0
                        for i in range(0,10):
                            if us_ct<int(allm[0][0]) and allm[i][1]!="":
                                frameUI = tk.Frame(win_c, bg='#dfffdb', width=200,highlightbackground='#3E4149', height=100)
                                frameUI.grid(column=int(i/5), row=i%5,padx=0.5,pady=0.5)
                                frames.append(frameUI)
                                L1=tk.Label(frames[i], text=allm[i][1],font=("Courier",12),bg='#dfffdb', height=2,justify=tk.CENTER)
                                L1.place(x=3, y=3)
                                L2=tk.Label(frames[i],bg='#dfffdb', height=70)
                                L2.place(x=65, y=18)
                                Ls.append(L1)
                                L1s.append(L2)
                                us_ct+=1
                            else:
                                frameUI = tk.Frame(win_c, bg='gray', width=200,highlightbackground='#3E4149', height=100)
                                frameUI.grid(column=int(i/5), row=i%5,padx=0.5,pady=0.5)
                                frames.append(frameUI)
                                L1=tk.Label(frames[i],bg='gray', height=80)
                                L1.place(x=60, y=30)
                                Ls.append(L1)
                                L1s.append(L1)
                                L1s[i].config(image=tk_img6)
                    user_i=int(allm[0][0])
                    us_ct=0
                    for i in range(0,7):
                        mc[i]=0
                    for i in range(0,10):
                        if us_ct<user_i:
                            if(i<len(allm) and allm[i][1]==""):
                                frames[i].config(bg='gray')
                                L1s[i].place(x=60, y=30)
                                Ls[i].config(bg='gray',text="")
                                L1s[i].config(image=tk_img6,height=80)
                                L1s[i].config(bg='gray')
                            elif(i<len(allm) and allm[i][2]=="mood: happy"):
                                L1s[i].config(image=tk_img1)
                                us_ct+=1
                                mc[0]+=1
                            elif(i<len(allm) and allm[i][2]=="mood: sad"):
                                L1s[i].config(image=tk_img2)
                                us_ct+=1
                                mc[1]+=1
                            elif(i<len(allm) and allm[i][2]=="mood: normal"):
                                L1s[i].config(image=tk_img4)
                                us_ct+=1
                                mc[3]+=1
                            elif(i<len(allm) and allm[i][2]=="mood: angry"):
                                L1s[i].config(image=tk_img3)
                                us_ct+=1
                                mc[2]+=1
                            elif(i<len(allm) and allm[i][2]=="mood: surprise"):
                                L1s[i].config(image=tk_img5)
                                us_ct+=1
                                mc[4]+=1
                            elif(i<len(allm) and allm[i][2]=="mood: disgust"):
                                L1s[i].config(image=tk_img7)
                                us_ct+=1
                                mc[5]+=1
                            elif(i<len(allm) and allm[i][2]=="mood: fear"):
                                L1s[i].config(image=tk_img8)
                                us_ct+=1
                                mc[6]+=1
                        else:
                                frames[i].config(bg='gray')
                                L1s[i].place(x=60, y=30)
                                Ls[i].config(bg='gray',text="")
                                L1s[i].config(image=tk_img6,height=80)
                                L1s[i].config(bg='gray')
                                Lau.config(text="觀眾: "+chr(user_i+48))
                    for i in range(0,7):
                        ms[i].config(text=chr(48+mc[i]))
                    
                    if qt1==1:
                        wt=1
                        break
                    timer=0
                Faces.clear()
                timer+=1
            cv2.putText(frame,prediction,(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255))
            cv2.imshow('MAE-DFER',frame)


            if cv2.waitKey(1) & 0xff == ord('q'):
               break
    frames=[]
    
    for i in range(0,10):
        frame = tk.Frame(win_c, bg='#dfffdb', width=200,highlightbackground='#3E4149', height=100)
        frame.grid(column=int(i/5), row=i%5,padx=0.5,pady=0.5)
        frames.append(frame)
    threading.Thread(target=UI).start()
    win_c.protocol("WM_DELETE_WINDOW",lambda: del_c(ser_num,user_n))
    def destroy(ser_num,user_n,name):
        global qt1
        global wt
        qt1=1
        while(wt==0):
            cv2.waitKey(1000)
        win_c.destroy()
        back(ser_num,user_n,name)
    global frame1
    frame1 = tk.Frame(win_c, bg='#1d551A', width=400, height=505)
    frame1.place(x=404,y=2)
    B_bk = tk.Button(frame1,text="返回",bg='#b4E2C5',font=("思源黑體",10),command=lambda: destroy(ser_num,user_n,name))
    B_bk.place(x=350,y=450)
    win_c.mainloop()

def record(ser_num,name):
    current_time = datetime.now()
    allus=worksheet.get_values(chr(66+ser_num*3)+':'+chr(67+ser_num*3))
    print(allus[0][0])
    for i in range(0,10):
       if i<int(allus[0][0]):
           if allus[i][1]=="":
               user_n=i
               break
       else:
           user_n=i
           break
    win_c = tk.Tk()
    win_c.title('Server')
    win_c.geometry('810x512')
    win_c.configure(bg='black')
    win_c.resizable(False, False)
    img1 = Image.open('./happy.png')
    tk_img1 = ImageTk.PhotoImage(img1)
    img2 = Image.open('./sad.png')
    tk_img2 = ImageTk.PhotoImage(img2)
    img3 = Image.open('./angry.png')
    tk_img3 = ImageTk.PhotoImage(img3)
    img4 = Image.open('./normal.png')
    tk_img4 = ImageTk.PhotoImage(img4)
    img5 = Image.open('./surprise.png')
    tk_img5 = ImageTk.PhotoImage(img5)
    img6 = Image.open('./avator.png')
    tk_img6 = ImageTk.PhotoImage(img6)
    img7 = Image.open('./disgust.png')
    tk_img7 = ImageTk.PhotoImage(img7)
    img8 = Image.open('./fear.png')
    tk_img8 = ImageTk.PhotoImage(img8)
    def UIs():
        global qt2
        global wt2
        qt2=0
        wt2=0
        mood=""
        user_i=user_n
        Ls=[]
        L1s=[]
        moods=[0,0,0,0,0,0,0]
        timer=0
        l_mood=-1
        alln=worksheet.col_values(3*(ser_num+1))
        us_ct=0
        for i in range(0,10):
            if us_ct<user_i and alln[i]!="":
                Lname=tk.Label(frames[i], text=alln[i],font=("Courier",12),bg='#dfffdb', height=2,justify=tk.CENTER)
                Lname.place(x=3, y=3)
                Ls.append(Lname)
                L1=tk.Label(frames[i],bg='#dfffdb', height=70)
                L1.place(x=65, y=18)
                us_ct+=1
                L1s.append(L1)
            else:
                frames[i].configure(bg='gray')
                Lname=tk.Label(frames[i], text="",font=("Courier",12),bg='gray', height=2,justify=tk.CENTER)
                Lname.place(x=3, y=3)
                Ls.append(Lname)
                L1=tk.Label(frames[i],bg='gray', height=80)
                L1.place(x=60, y=30)
                L1s.append(L1)
                L1s[i].config(image=tk_img6)
        Lau=tk.Label(frame,text="觀眾: "+chr(48+user_i), bg='#1d551A',font=("隸書",20),fg='white')
        Lau.place(x=10,y=10)
        m1=tk.Label(frame,image=tk_img1, bg='#1d551A',height=50)
        m1.place(x=10,y=100)
        m2=tk.Label(frame,image=tk_img2, bg='#1d551A',height=50)
        m2.place(x=10,y=190)
        m3=tk.Label(frame,image=tk_img3, bg='#1d551A',height=50)
        m3.place(x=10,y=280)
        m4=tk.Label(frame,image=tk_img4, bg='#1d551A',height=50)
        m4.place(x=10,y=370)
        m5=tk.Label(frame,image=tk_img5, bg='#1d551A',height=50)
        m5.place(x=210,y=100)
        m6=tk.Label(frame,image=tk_img7, bg='#1d551A',height=50)
        m6.place(x=207,y=190)
        m7=tk.Label(frame,image=tk_img8, bg='#1d551A',height=50)
        m7.place(x=205,y=280)
        ms=[]
        mc=[]
        timer=0
        timer_c=0
        b_mood=0
        g_mood=0
        for i in range(0,7):
                m=tk.Label(frame,text=chr(48), bg='#1d551A',font=("隸書",20),fg='white')
                m.place(x=int(i/4)*200+120,y=90*((i%4)+1)+15)
                ms.append(m)
                mc.append(0)
        formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
        worksheet.update("A"+chr(66+int(ser_num)*3)+str(3),formatted_time)
        while True:
            if qt2==1:
                wt2=1
                break
            allm=worksheet.get_values(chr(66+ser_num*3)+':'+chr(68+ser_num*3))
            if int(allm[0][0])>user_i:
                        L1s.clear()
                        Ls.clear()
                        frames.clear()
                        us_ct=0
                        for i in range(0,10):
                            if us_ct<int(allm[0][0]) and allm[i][1]!="":
                                frameUI = tk.Frame(win_c, bg='#dfffdb', width=200,highlightbackground='#3E4149', height=100)
                                frameUI.grid(column=int(i/5), row=i%5,padx=0.5,pady=0.5)
                                frames.append(frameUI)
                                L1=tk.Label(frames[i], text=allm[i][1],font=("Courier",12),bg='#dfffdb', height=2,justify=tk.CENTER)
                                L1.place(x=3, y=3)
                                L2=tk.Label(frames[i],bg='#dfffdb', height=70)
                                L2.place(x=65, y=18)
                                Ls.append(L1)
                                L1s.append(L2)
                                us_ct+=1
                            else:
                                frameUI = tk.Frame(win_c, bg='gray', width=200,highlightbackground='#3E4149', height=100)
                                frameUI.grid(column=int(i/5), row=i%5,padx=0.5,pady=0.5)
                                frames.append(frameUI)
                                L1=tk.Label(frames[i],bg='gray', height=80)
                                L1.place(x=60, y=30)
                                Ls.append(L1)
                                L1s.append(L1)
                                L1s[i].config(image=tk_img6)
            user_i=int(allm[0][0])
            us_ct=0
            for i in range(0,7):
                mc[i]=0
            if(timer>=4):
                sum_p=0
                maxi=0
                maxm=0
                for i in range(0,7):
                    sum_p+=moods[i]
                    if moods[i]>=maxi:
                        maxi=moods[i]
                        maxm=i
                    
                print(maxi)
                worksheet.update("A"+chr(67+int(ser_num)*3)+str(timer_c+1),allm[0][0])
                #list1=str(int(maxi/sum_p*100))+','+str(int(g_mood/sum_p*100))+','+str(int(b_mood/sum_p*100))
                list1=""
                for i in range(0,7):
                    if i!=6:
                        list1+=str(round(moods[i]/5))+','
                        moods[i]=0
                    else:
                        list1+=str(round(moods[i]/5))
                        moods[i]=0
                worksheet.update("A"+chr(68+int(ser_num)*3)+str(timer_c+1),list1)
                timer=0
                timer_c+=1
                b_mood=0
                g_mood=0
                worksheet.update("A"+chr(66+int(ser_num)*3)+"2",timer_c)
            for i in range(0,10):
                if us_ct<user_i:
                    if(i<len(allm) and allm[i][1]==""):
                        frames[i].config(bg='gray')
                        L1s[i].place(x=60, y=30)
                        Ls[i].config(bg='gray',text="")
                        L1s[i].config(image=tk_img6,height=80)
                        L1s[i].config(bg='gray')
                    elif(i<len(allm) and allm[i][2]=="mood: happy"):
                        L1s[i].config(image=tk_img1)
                        us_ct+=1
                        mc[0]+=1
                        moods[0]+=1
                        g_mood+=1
                    elif(i<len(allm) and allm[i][2]=="mood: sad"):
                        L1s[i].config(image=tk_img2)
                        us_ct+=1
                        mc[1]+=1
                        moods[1]+=1
                        b_mood+=1
                    elif(i<len(allm) and allm[i][2]=="mood: normal"):
                        L1s[i].config(image=tk_img4)
                        us_ct+=1
                        mc[3]+=1
                        moods[2]+=1
                    elif(i<len(allm) and allm[i][2]=="mood: angry"):
                        L1s[i].config(image=tk_img3)
                        us_ct+=1
                        mc[2]+=1
                        moods[3]+=1
                        b_mood+=1
                    elif(i<len(allm) and allm[i][2]=="mood: surprise"):
                        L1s[i].config(image=tk_img5)
                        us_ct+=1
                        mc[4]+=1
                        moods[4]+=1
                        g_mood+=1
                    elif(i<len(allm) and allm[i][2]=="mood: disgust"):
                        L1s[i].config(image=tk_img7)
                        us_ct+=1
                        mc[5]+=1
                        moods[5]+=1
                        b_mood+=1
                    elif(i<len(allm) and allm[i][2]=="mood: fear"):
                        L1s[i].config(image=tk_img8)
                        us_ct+=1
                        mc[6]+=1
                        moods[6]+=1
                        b_mood+=1
                else:
                    frames[i].config(bg='gray')
                    L1s[i].place(x=60, y=30)
                    Ls[i].config(bg='gray',text="")
                    L1s[i].config(image=tk_img6,height=80)
                    L1s[i].config(bg='gray')
            Lau.config(text="觀眾: "+chr(user_i+48))
            for i in range(0,7):
                ms[i].config(text=chr(48+mc[i]))
            if qt2==1:
                wt2=1
                break
            timer+=1
            cv2.waitKey(2000)
    frames=[]
    
    for i in range(0,10):
        frame = tk.Frame(win_c, bg='#dfffdb', width=200,highlightbackground='#3E4149', height=100)
        frame.grid(column=int(i/5), row=i%5,padx=0.5,pady=0.5)
        frames.append(frame)
    frame = tk.Frame(win_c, bg='#1d551A', width=400, height=505)
    frame.place(x=404,y=2)
    alln=worksheet.col_values(3*(ser_num+1))
    threading.Thread(target=UIs).start()
    def destroy():
        global qt2
        global wt2
        qt2=1
        while(wt2==0):
            cv2.waitKey(1000)
        win_c.destroy()
        ChooseServer(1,name)
    B_bk = tk.Button(frame,text="返回",bg='#b4E2C5',font=("思源黑體",10),command=destroy)
    B_bk.place(x=350,y=450)
    win_c.mainloop()

def name(ch,n):
    def getname(c,n):
        name=test1.get(1.0,"end")
        win_n.destroy()
        if(c==0):
            client(name,n)
        else:
            server(name)
    win_n = tk.Tk()
    win_n.title('Name')
    win_n.geometry('380x400+100+50')
    win_n.resizable(False, False)
    win_n.configure(bg='#d0eff1')
    test = tk.Label(text="請輸入暱稱",bg='#d0eff1',font=("思源黑體",20))
    test.place(x=110,y=30)
    test1 = tk.Text(win_n, height=0.5,width=15)
    test1.place(x=120,y=110)
    B1 = tk.Button(text="確定",bg='#8Ce2f3',font=("思源黑體",13),relief='ridge',command=lambda: getname(ch,n))
    B1.place(x=150,y=160)
    win_n.mainloop()
    
def report(i_1,name):
    global md
    md=0
    global rg
    rg=18
    global re_c
    re_c=0
    global bag
    bag=1
    win = tk.Tk()
    num_h=worksheet.acell('A1').value
    win.title("Report")
    win.geometry('1000x700+300+20')
    win.configure(bg='#bbffb7')
    win.resizable(False, False)
    s_time=worksheet.acell('A'+chr(66+i_1*3)+"3").value
    Label1 = tk.Label(bg='#bbffb7',height=1,text="直播時間： "+s_time,font=('思源黑體',16)).place(x=200,y=10)
    if int(num_h)!=0:
        def des(i,re_c,name):
            win.destroy()
            report(i,name)
        def des1(name):
            win.destroy()
            ChooseServer(0,name)
        tk_img=[]
        img1 = Image.open('./happy.png')
        re_img1 = img1.resize((30, 30))
        tk_img.append(ImageTk.PhotoImage(re_img1))
        img2 = Image.open('./sad.png')
        re_img2 = img2.resize((30, 30))
        tk_img.append(ImageTk.PhotoImage(re_img2))
        img3 = Image.open('./normal.png')
        re_img3 = img3.resize((30, 30))
        tk_img.append(ImageTk.PhotoImage(re_img3))
        img4 = Image.open('./angry.png')
        re_img4 = img4.resize((30, 30))
        tk_img.append(ImageTk.PhotoImage(re_img4))
        img5 = Image.open('./surprise.png')
        re_img5 = img5.resize((30, 30))
        tk_img.append(ImageTk.PhotoImage(re_img5))
        img6 = Image.open('./disgust.png')
        re_img6 = img6.resize((33, 33))
        tk_img.append(ImageTk.PhotoImage(re_img6))
        img7 = Image.open('./fear1.png')
        re_img7 = img7.resize((40, 40))
        tk_img.append(ImageTk.PhotoImage(re_img7))
        colors = ['red', 'blue', 'yellow', 'green', 'orange', 'purple','black']
        moods=['happy', 'sad', 'normal', 'angry', 'surprise', 'disgust','fear']
        space="                    "
        for i in range(0,7):
            Label1 = tk.Label(bg=colors[i],height=1,text=space).place(x=870,y=55+int(i)*60)
            Label1 = tk.Label(bg='#bbffb7',image=tk_img[i]).place(x=800,y=50+int(i)*60)
        num=worksheet.acell('A'+chr(66+i_1*3)+"2").value
        if re_c<0:
            re_c=int(int(num)/rg)
        elif re_c>int(int(num)/rg):
            re_c=0
        mood_3=np.zeros((7, rg))
        data=worksheet.get_values("A"+chr(67+i_1*3)+':'+"A"+chr(68+i_1*3))
        max_p=0
        for i in range(0,rg):
            if(re_c*rg+i>=int(num)):
                break
            if(int(data[re_c*rg+i][0])>max_p):
               max_p=int(data[re_c*rg+i][0])
        for i in range(0,rg):
            j1=0
            if(re_c*rg+i>=int(num)):
                    break
            for k in range(len(data[re_c*rg+i][1])):
                if(data[re_c*rg+i][1][k]!=','):
                    mood_3[j1][i]=10*mood_3[j1][i]+int(data[re_c*rg+i][1][k])
                else:
                    j1+=1
        def ntp():
            global re_c
            global box1
            print(box1.current())
            re_c=box1.current()
            update_plot(0,i_1)
        def update_plot(plus,i_1):
            data=worksheet.get_values("A"+chr(67+i_1*3)+':'+"A"+chr(68+i_1*3))
            num=worksheet.acell('A'+chr(66+i_1*3)+"2").value
            global md
            global bag
            global rg
            global re_c
            global Btr1
            global box1
            if plus==1:
                md=(md+1)%4
                re_c=0
            max_p=0
            if md==0:
                rg=18
                bag=1
                Btr1.config(text="1m")
            elif md==1:
                rg=10
                bag=6
                Btr1.config(text="10m")
            elif md==2:
                rg=18
                bag=60
                Btr1.config(text="30m")
            elif md==3:
                rg=10
                bag=180
                Btr1.config(text="1s")
            mood_4=np.zeros((7, rg*bag))
            vls1=[]
            for i in range(0,int(int(num)/(rg*bag))+1):
                vls1.append(str(i+1))
            box1['values']=vls1
            # 生成新的随机数据
            for i in range(0,rg):
                if(re_c*rg*bag+i>=int(num)):
                    break
                if(int(data[re_c*rg*bag+i][0])>max_p):
                   max_p=int(data[re_c*rg*bag+i][0])
            for i in range(0,rg*bag):
                j1=0
                if(re_c*rg*bag+i>=int(num)):
                    break
                for k in range(len(data[re_c*rg*bag+i][1])):
                    if(data[re_c*rg*bag+i][1][k]!=','):
                        mood_4[j1][i]=10*mood_4[j1][i]+int(data[re_c*rg*bag+i][1][k])
                    else:
                        j1+=1
            mood_5=np.zeros((7, rg))
            for i in range(0,rg):
                for j in range(bag):
                    for k in range(7):
                        mood_5[k][i]+=mood_4[k][i*bag+j]
            for i in range(0,rg):
                for j in range(0,7):
                    mood_5[j][i]=mood_5[j][i]/bag
           # 更新图形
            ax.clear()
            plt.xlabel('t', fontsize=13)
            plt.ylabel('num', fontsize=13)
            x_ticks_positions = np.arange(0, rg, 1)
            x_ticks_labels= np.arange(0, rg, 1)
            x_ticks_labels=x_ticks_labels.astype(str)
            for i in x_ticks_positions:
                if i%2==0:
                    x_ticks_labels[i]=str(int((re_c*rg+i)*bag/6))+':'+str((re_c*rg+i)*bag*10%60)
                else:
                    x_ticks_labels[i]=''
            plt.xticks(x_ticks_positions, x_ticks_labels)
            for i in range(0,mood_5.shape[0]):
                ax.plot(mood_5[i], color=colors[i],  linestyle='solid',marker = 'o',markersize = 2)
            canvas.draw()

        L1=tk.Label(text="選擇報告",bg='#bbffb7',font=("思源黑體",16), height=2).place(x=50, y=500)
        L1=tk.Label(text="時間單位",bg='#bbffb7',font=("思源黑體",16), height=2).place(x=50, y=550)
        L1=tk.Label(text="選擇頁數",bg='#bbffb7',font=("思源黑體",16), height=2).place(x=50, y=600)
        box = ttk.Combobox(win)
        box.place(x=150,y=515)
        r_n=worksheet.col_values(1)
        vls=[]
        for i in range(0,int(r_n[0])):
            vls.append(r_n[i+1])
        box['values']=vls
        global box1
        box1 = ttk.Combobox(win)
        box1.place(x=150,y=615)
        vls1=[]
        for i in range(0,int(int(num)/(rg*bag))+1):
            vls1.append(str(i+1))
        box1['values']=vls1
        b1=tk.Button(win,text="choose",bg='#002fff',font=("思源黑體",11),fg='white', command=lambda: destroywin(box.current())).place(x=150, y=150)    
        fig, ax = plt.subplots(figsize=(7, 4))
        x_ticks_positions = np.arange(0, rg, 1)
        x_ticks_labels= np.arange(0, rg, 1)
        x_ticks_labels=x_ticks_labels.astype(str)
        for i in x_ticks_positions:
            if i%2==0:
                x_ticks_labels[i]=str(int((re_c*rg+i)/6))+':'+str((re_c*rg+i)*10%60)
            else:
                x_ticks_labels[i]=''
        plt.xticks(x_ticks_positions, x_ticks_labels)
        for i in range(0,mood_3.shape[0]):
            ax.plot(mood_3[i], color=colors[i],  linestyle='solid',marker = 'o',markersize = 2)
        plt.xlim(-0.5,int(rg))
        plt.ylim(-0.5,max_p+1)
        plt.title(vls[i_1], fontsize=10)
        plt.xlabel('t', fontsize=13)
        plt.ylabel('num', fontsize=13)
        
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.place(x=50,y=50)
        # 添加退出按钮
        Btr = tk.Button(text="確定",fg='black',font=('思源黑體',10),bg='#b4E2C5',relief='groove',command=ntp).place(x=330,y=615)
        Btr = tk.Button(text="確定",fg='black',font=('思源黑體',10),bg='#b4E2C5',relief='groove',command=lambda:des((i_1+box.current())%int(num_h),0,name)).place(x=330,y=512)
        global Btr1
        Btr1 = tk.Button(text="10s",fg='black',font=('思源黑體',10),bg='#b4E2C5',relief='groove',command=lambda:update_plot(1,i_1))
        Btr1.place(x=160,y=560)
    win.mainloop()
def ChooseServer(isbk,name):
    sern=worksheet.col_values(1)
    server_n=int(sern[0])
    vls=[]
    def destroywin(snum,b):
        win.destroy()
        if b==0:
            client(name,snum)
        else:
            server(name)
    def destroy1(name):
        win.destroy()
        report(0,name)
    for i in range(0,server_n):
        vls.append(sern[i+1])
    win = tk.Tk()
    win.title('Choose Server')
    win.geometry('380x420+100+50')
    win.configure(bg='#bbffb7')
    img1 = Image.open('./eye1.png')
    tk_img1 = ImageTk.PhotoImage(img1)
    L1=tk.Label(text="選擇觀看頻道",bg='#bbffb7',font=("思源黑體",20), height=2).place(x=95, y=5)
    Btr = tk.Button(text="統計",fg='black',font=('思源黑體',15),bg='#b4E2C5',relief='groove',command=lambda:destroy1(name)).place(x=300,y=20)
    L1=[]
    L2=[]
    L3=[]
    B=[]
    Bs=[]
    F=[]
    for i in range(0,8):
        frame = tk.Frame(win, bg='#dfffdb',highlightbackground='#3E4149', height=60,width=150)
        frame.place(x=int(i/4)*200+10,y=75+(i%4)*70)
        F.append(frame)
    for i in range(0,8): 
        L = tk.Label(F[i],text="",font=12,bg='#dfffdb', height = 2, width = 10)
        L.place(x=5,y=5)
        L0=tk.Label(F[i],text=" ",font=8,bg='#dfffdb', height = 1, width = 1)
        L0.place(x=120,y=6)
        Li=tk.Label(F[i],text=" ",font=8,bg='#dfffdb', height = 15,image=tk_img1)
        Li.place(x=90,y=6)
        B1 = tk.Button(F[i],text="加入",fg='black',font=('Arial',7),bg='#b4E2C5',state=tk.DISABLED, height = 1, width = 3,relief='groove')
        B1.place(x=115,y=33)
        B.append(B1)
        #B2 = tk.Button(F[i],text="紀錄",fg='black',font=('Arial',7),bg='#b4E2C5',state=tk.DISABLED, height = 1, width = 3,relief='groove')
        #B2.place(x=65,y=33)
        #Bs.append(B2)
        L1.append(L)
        L2.append(L0)
        L3.append(Li)
    for i in range(0,server_n):
       cn=worksheet.acell(chr(66+i*3)+'1').value
       L2[i].config(text=cn,font=('Arial',8),fg='black')
       L1[i].config(text=sern[i+1],font=('Arial',12),anchor='w')
       B[i].config(state=tk.NORMAL,command=lambda i1=i:destroywin(i1,0))
       #Bs[i].config(state=tk.NORMAL,command=lambda i1=i:destroywin(i1,2))
    for i in range(server_n,8):
       B[i].config(text="新增",state=tk.NORMAL,command=lambda i1=i:destroywin(i1,1))
    win.mainloop()

def server(name):
    server=worksheet.acell('A1').value
    server_n=int(server)+1
    worksheet.update('A1', server_n)
    worksheet.update(chr(66+(server_n-1)*3)+'1', 0)
    worksheet.update('A'+chr(48+server_n+1), name)
    print(server_n)
    record(server_n-1,name)
def getname():
        name=test1.get(1.0,"end")
        window.destroy()
        ChooseServer(0,name)
window.configure(bg='black')
import random

"""
for i in range(1000):
    w=""
    x=0
    sum=0
    for j in range(7):
        x=random.randint(0,8-sum)
        sum+=x
        if(j!=6):
            w+=str(x)+','
        else:
            w+=str(x)
    print(w)
    worksheet.update("AE"+str(15+i),w)
    cv2.waitKey(1200)
"""
img1 = Image.open('./happy.png')
tk_img1 = ImageTk.PhotoImage(img1)
L1=tk.Label(text="MoodResonate",bg='black',font=("思源黑體",25,'bold'),fg='white').place(x=65, y=10)
L2=tk.Label(text="Stream",bg='black',font=("思源黑體",25,'bold'),fg='white').place(x=125, y=50)
L3=tk.Label(image=tk_img1,bg='black')
L3.place(x=155, y=110)
test1 = tk.Text(window, height=1,width=15)
test1.place(x=125,y=175)
B1 = tk.Button(text="Enter",bg='#b4E2C5',font=("思源黑體",17,'bold'),relief='ridge',fg='white',command=getname)
B1.place(x=145,y=210)
window.mainloop() 
