from fastapi import FastAPI

app = FastAPI()

@app.post("/api/")
async def analyze_task(task: dict):
    return {"message": "Task received", "task": task}
