from fastapi import FastAPI, Form, Request, Response, File, Depends, HTTPException, status
from fastapi.responses import RedirectResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.encoders import jsonable_encoder
import uvicorn
import os
import aiofiles
import json
import csv
from src.helper import llm_qa_gen_pipeline, llm_website_content_pipeline, get_website_chatbot_response


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/file_generate_qa", response_class=HTMLResponse)
async def file_upload(request: Request):
    """Render the file upload page."""
    return templates.TemplateResponse("file_generate_qa.html", {"request": request})


@app.get("/website_chatbot", response_class=HTMLResponse)
async def website_chatbot(request: Request):
    """Render the chatbot page."""
    return templates.TemplateResponse("website_chatbot.html", {"request": request})

@app.post("/generate_qa_from_upload")
async def generate_qa_from_upload(request: Request, pdf_file: bytes = File(), filename: str = Form(...)):
    base_folder = 'static/docs/'
    if not os.path.isdir(base_folder):
        os.mkdir(base_folder)
    pdf_filename = os.path.join(base_folder, filename)

    async with aiofiles.open(pdf_filename, 'wb') as f:
        await f.write(pdf_file)
 
    response_data = jsonable_encoder(json.dumps({"msg": 'success',"pdf_filename": pdf_filename}))
    res = Response(response_data)
    return res


def get_qa_csv(file_path):
    answer_generation_chain, ques_list = llm_qa_gen_pipeline(file_path)
    base_folder = 'static/output/'
    if not os.path.isdir(base_folder):
        os.mkdir(base_folder)
    output_file = base_folder+"QA.csv"
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Question", "Answer"])  # Writing the header row

        for question in ques_list:
            print("Question: ", question)
            answer = answer_generation_chain.invoke({"input": question})
            print("Answer: ", answer.get("answer"))
            print("--------------------------------------------------\n\n")

            # Save answer to CSV file
            csv_writer.writerow([question, answer.get("answer")])
    return output_file


@app.post("/analyze_prepare_qa")
async def analyze_prepare_qa(request: Request, pdf_filename: str = Form(...)):
    output_file = get_qa_csv(pdf_filename)
    response_data = jsonable_encoder(json.dumps({"output_file": output_file}))
    res = Response(response_data)
    return res

@app.post("/get_website_content")
async def get_website_content(request: Request):
    """Process the website URL and prepare embeddings."""
    data = await request.json()
    url = data.get("url")
    if not url:
        return JSONResponse({"success": False, "error": "Invalid URL"}, status_code=400)

    # Process the URL (scrape content, generate embeddings, etc.)
    try:
        success = llm_website_content_pipeline(url)  # Your custom logic for RAG
        if success:
            return {"success": True}
        else:
            return JSONResponse({"success": False, "error": "Failed to process URL"}, status_code=500)
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)
    
@app.post("/chat_with_website_content")
async def chat_with_website_content(request: Request):
    """Handle chatbot queries based on processed content."""
    data = await request.json()
    question = data.get("question")
    if not question:
        return JSONResponse({"answer": "Invalid question"}, status_code=400)

    try:
        # Get the response using RAG
        website_qa_chain = get_website_chatbot_response()  # Your custom RAG logic
        answer = website_qa_chain.invoke({"input": question})
        return {"answer": answer.get("answer")}
    except Exception as e:
        return JSONResponse({"answer": f"Error: {str(e)}"}, status_code=500)



if __name__ == "__main__":
    uvicorn.run("app:app", host='0.0.0.0', port=8080, reload=True)