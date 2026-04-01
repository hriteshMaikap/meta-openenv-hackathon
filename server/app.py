"""
server/app.py — FastAPI server wrapping the ETL environment.
One call to create_fastapi_app() creates all OpenEnv endpoints.
"""
import os
from environment import ETLEnvironment

try:
    from openenv.core.env_server import create_fastapi_app
    app = create_fastapi_app(
        lambda: ETLEnvironment(
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
            judge_model=os.environ.get("JUDGE_MODEL", "gpt-4o-mini"),
        )
    )
except ImportError:
    # Local dev fallback
    from fastapi import FastAPI
    app = FastAPI(title="ETL Pipeline Agent (dev mode)")
    
    _env = ETLEnvironment()
    
    @app.post("/reset")
    def reset(task_id: str = "easy", seed: int = 42):
        obs = _env.reset(task_id=task_id, seed=seed)
        return obs.model_dump()
    
    @app.post("/step")
    def step(tool: str, params: dict = {}, reasoning: str = ""):
        from models import ETLAction
        action = ETLAction(tool=tool, params=params, reasoning=reasoning)
        obs, reward, done, info = _env.step(action)
        return {"observation": obs.model_dump(), "reward": reward, "done": done, "info": info}
    
    @app.get("/state")
    def state():
        return _env.state.model_dump()
    
    @app.get("/health")
    def health():
        return {"status": "ok"}