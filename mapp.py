import os
import yaml
import streamlit as st
import helper
import numpy as np
import torch 
from torchvision import models, transforms
from PIL import Image
from types import SimpleNamespace as SN
from  u2net import U2NET
from cn_net import cnnet
from pama_net import Net

cls_thr =0.2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

seg_weights = 'c:/make-charactor/u2net.pth'
cls_weights = 'c:/make-charactor/cn_best.pth'

breed_sheet = 'c:/make-charactor/class_sheet.yaml'
charactor_dir = 'c:/make-charactor/char_data/'
results_dir = 'c:/make-charactor/results/'

trargs = SN()
trargs.pretrained = True
trargs.requires_grad = False
trargs.training = False
trargs.encoder_weights = 'c:/make-charactor/pama-check/encoder.pth'
trargs.decoder_weights = 'c:/make-charactor/pama-check/decoder.pth'
trargs.align1_weights = 'c:/make-charactor/pama-check/PAMA1.pth'
trargs.align2_weights = 'c:/make-charactor/pama-check/PAMA2.pth'
trargs.align3_weights = 'c:/make-charactor/pama-check/PAMA3.pth'

seg_input_size = 320
segpre1 = transforms.Compose([
                transforms.Resize(seg_input_size),
                transforms.CenterCrop(seg_input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
segpre2 = transforms.Compose([
                transforms.Resize(seg_input_size),
                transforms.CenterCrop(seg_input_size),
                transforms.ToTensor()])

cls_input_size = 384
clspre = transforms.Compose([
                    transforms.Resize(cls_input_size),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

trs_input_size = 512
trsprep = transforms.Compose([
                transforms.Resize(trs_input_size),
                transforms.ToTensor()])
trspret = transforms.Resize(trs_input_size)


# UI layout
st.set_page_config(page_title="Pet Breed Classification")
st.markdown(
        body=helper.UI.css,
        unsafe_allow_html=True,
)
# Sidebar
st.sidebar.markdown(helper.UI.about_block, unsafe_allow_html=True)

# Title
st.header("Pet Breed Classification")

# File uploader
upload_cell, preview_cell = st.columns([12, 1])
query = upload_cell.file_uploader("Image of your pet")


''' segment model '''

def predmask(pred, thres=0.05):
    ma = torch.max(pred)
    mi = torch.min(pred)
    dn = (pred-mi)/(ma-mi)
    mask = torch.ge(dn, thres)
    return mask

def segment(inqry, bweights=seg_weights, device=device):
    image = Image.open(inqry)
    img_tensor = segpre1(image).unsqueeze(0)
    imgt = segpre2(image)

    model = U2NET(3,1)
    state = torch.load(bweights, map_location='cpu')
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()

    img_tensor = img_tensor.to(device)
    d1,d2,d3,d4,d5,d6,d7 = model(img_tensor)        
    pred = d1[:,0,:,:]
    mask = predmask(pred.cpu())
    segimg = imgt*mask
    return segimg, image    # tensor, PIL
    

''' classifer model '''

def get_char_img(bred, sheet=breed_sheet, chdir=charactor_dir):
    with open(sheet, 'r') as stream:
        breeds = yaml.load(stream, Loader=yaml.FullLoader)
    nbrd = breeds[bred]
    img_file = os.path.join(chdir, nbrd+'.jpg')
    return Image.open(img_file)


def classify(inqry, thres=cls_thr, weights=cls_weights, device=device):
    segimg, image = segment(inqry)  # tensor, PIL
    simg = clspre(segimg)
    sgimg = simg.unsqueeze(0).to(device)

    model = cnnet()
    model.load_state_dict(torch.load(weights, map_location=device))
    model = model.to(device)
    model.eval()

    output = torch.nn.Softmax(dim=1)(model(sgimg))
    prob, label = torch.max(output, 1)
    prob = prob.cpu().squeeze().tolist()
    label = label.cpu().squeeze().tolist()
    if prob < thres:
        if label <= 11:
            lbl = 37
        else:
            lbl = 38
    else:
        lbl = label
    char_img = get_char_img(lbl)
    print('label:', lbl, 'prob:', prob)
    return char_img, segimg, image   # PIL, tensor, PIL


'''style transfer model'''

def transfer(inqry, args=trargs, device=device):
    simg, rimg, qimage = classify(inqry)  # PIL, Tensor, PIL
    source = trsprep(simg).unsqueeze(0)
    refer = trspret(rimg).unsqueeze(0)
    source = source.to(device)
    refer = refer.to(device)

    model = Net(args)
    model = model.to(device)
    model.eval()

    trans_img = model(source, refer)
    return trans_img, qimage          # tensor, PIL

''' images view'''
def show_save_images(timg, qimg, inqry):
    ttimg = timg.clamp_(0.0, 1.0).permute(0, 2, 3, 1).squeeze(0).numpy(force=True)
    ntim = np.array(ttimg*255, dtype=np.uint8)    
    nqim = np.array(qimg, dtype=np.uint8)

    name = inqry.name
    img_name = name.split(os.sep)[-1]
    trsimg = Image.fromarray(ntim)
    chrimg = trsimg.resize((260,260))
    nchr = np.array(chrimg, dtype=np.uint8)

    chrimg.save(results_dir+'transfered_'+img_name)
    qimg.save(results_dir+'query_'+img_name)

    image = [nqim, nchr]
    capt = ["Query Image", "generated CHARACTOR"]
    st.image(image, caption=capt, clamp=True)


# If file is uploaded
if query:
    # if clicked on 'classify' button
    if st.button(label="Generate"):
        # read image and process it for the model
        trsimg, qimage = transfer(query)        
        show_save_images(trsimg, qimage, query)