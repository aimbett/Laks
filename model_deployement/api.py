#!/usr/bin/python
from sklearn.externals import joblib
from m10_model_deployment import predict_image
from fastapi import FastAPI
import uvicorn

#app = Flask(__name__)
app = FastAPI(title='Laks API')

api = Api(
    app, 
    version='1.0', 
    title='Laks Lice Classificator API',
    description='Lice Classificator API')

ns = api.namespace('predict', 
     description='Lice Classifier')
    
parser = api.parser()

parser.add_argument(
    'URL', 
    type=str, 
    required=True, 
    help='URL to be analyzed', 
    location='args')

resource_fields = api.model('Resource', {
    'result': fields.String,
})



@app.get('/index')
async def hello_world():
    return "hello world"

def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image


@app.post("/predict/image")
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png", "PNG")
    if not extension:
        return "Image must be jpg or png format!"
    image = read_imagefile(await file.read())
    prediction = predict_image(image)
    return {"result": prediction}, 200


    
    
if __name__ == '__main__':
    #app.run(debug=True, use_reloader=False, host='0.0.0.0', port=8888)
    uvicorn.run(app, debug=True, use_reloader=False, host='0.0.0.0', port=8888)