from fastapi import FastAPI ,Form, Request 
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM 
import uvicorn
import torch

app = FastAPI()

@app.on_event("startup")
def my_model():
    app.state.model = AutoModelForCausalLM.from_pretrained("gpt2")
    app.state.tokenizer = AutoTokenizer.from_pretrained("gpt2")
@app.post("/generate")
def report_generator(
                     request: Request,
                     patient_name : str = Form(...),
                     patient_age : int = Form(...),
                     gender : str = Form(...),
                     history : str = Form(...),
                     findings : str = Form(...),
                     doctors_name : str = Form(...),
                     doctors_title : str = Form(...),
                     ref_no : str = Form(...)
                    ):
    example = """
10/18/2025 8:20:58 PM

NAME: Mal. Yakubu Abdullahi 
Age: Ad
Sex:  M
Ref No:0008
Clinical Details:
US:08

REAL TIME ABDOMINAL AND PELVIC ULTRASONOGRAPHY EXAM: REVEALED:

The liver is normal in size and smooth in outline but demonstrates increased parenchymal echogenicity with attenuation of the posterior beam, consistent with moderate fatty infiltration (hepatic steatosis). No focal hepatic lesion or intrahepatic biliary dilatation is seen. The intrahepatic vessels are normally visualized, and the portal vein shows normal flow direction and caliber.

The gallbladder is well distended with normal wall thickness. No gallstones, sludge, or pericholecystic fluid collection is seen. The common bile duct measures within normal limits and is not dilated.
The visualized portions of the pancreas appear normal in size and echotexture. No focal lesion or peripancreatic fluid collection is observed.

The spleen is normal in size and echotexture, with a homogeneous parenchymal pattern. No focal splenic lesion is seen.

Both kidneys are normal in size, outline, and cortical echotexture. The corticomedullary differentiation is preserved. No hydronephrosis, renal calculus, or focal renal mass is seen.

The urinary bladder is adequately distended with smooth contour and normal wall thickness. No intraluminal mass, stone, or diverticulum is seen.

The prostate gland is enlarged, measuring approximately 3.69 cm (AP) × 4.53 cm (transverse) × 4.76 cm (craniocaudal), with a calculated volume of 41.67 cm³ (normal ≤ 30 cm³). The echotexture appears mildly heterogeneous without focal nodules or calcifications. The median lobe is slightly prominent but does not significantly indent the bladder base. Findings are in keeping with benign prostatic enlargement (BPH).
Impression: 1. Moderate fatty liver (hepatic steatosis). 2. Enlarged prostate gland (volume 41.67 cm³) — features suggestive of benign prostatic enlargement (BPH).


Doctor: Sagir Usman 		
Title: Medical ultrasound specialist
                            """
    model = request.app.state.model
    tokenizer = request.app.state.tokenizer
    re = {"patient name":patient_name,
         "patient age":patient_age,
         "patient gender":gender,
         "history": history ,
         "findings":findings,
         "doctors name":doctors_name,
         "doctors title":doctors_title,
         "Rf no":ref_no}
    prompt = f"""
    you are an ultrasound report generator your task are:
    1) generate an ultrasound report in this example format {example}
    2) this is the data that you will use to make the report in the format{re}
    """
    inputs = tokenizer(prompt, return_tensors = "pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, 
                                 max_length = 800,
                             	top_p = 0.9,
                             	temperature = 0.5)
    report = tokenizer.decode(outputs[0],skip_special_tokens = True)
    return {"report":report}


if __name__ == "__main__":
    uvicorn.run(app,host = "0.0.0.0", port = 8000)