import os
import io
import json
import torch
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from model import MobileNetV2

app=Flask(__name__)
CORS(app)

weights_path = "../MobileNet/MobileNet_v2.pth"
class_json_path = "../MobileNet/class_indices.json"
assert os.path.exists(weights_path), "weights path does not exist..."
assert os.path.exists(class_json_path), "class json path does not exist..."

devive=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(devive)

model=MobileNetV2(num_classes=5).to(devive)
model.load_state_dict(torch.load(weights_path,map_location=devive))

model.eval()

json_file=open(class_json_path,'rb')
class_indict=json.load(json_file)

def transform_image(image_bytes):
    my_transforms=transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])
    image=Image.open(io.BytesIO(image_bytes))

    if image.mode !='RGB':
        raise ValueError("input file does not RGB image...")
    return my_transforms(image).unsqueeze(0).to(devive)

def get_prediction(image_bytes):
    try:
        tensor=transform_image(image_bytes)
        outputs=torch.softmax(model.forward(tensor).squeeze(),dim=0)
        prediction=outputs.detach().cpu().numpy()
        template = "class:{:<15} probability:{:.3f}"
        index_pre=[(class_indict[str(index)],float(p)) for index,p in enumerate(prediction)]
        index_pre.sort(key=lambda x:x[1],reverse=True)
        text=[template.format(k,v) for k,v in index_pre]
        return_info={'result':text}
    except Exception as e:
        return_info={'result':[str(e)]}
    return return_info


@app.route("/predict",methods=["POST"])
@torch.no_grad()
def predict():
    image=request.files["file"]
    img_bytes=image.read()
    info=get_prediction(image_bytes=img_bytes)
    return jsonify(info)

@app.route("/",methods=["GET","POST"])
def root():
    return render_template("up.html")

if __name__=='__main__':
    app.run(host='0.0.0.0',port=5001)