## app/models.py

from typing import List, Optional
from pydantic import BaseModel

class Panel(BaseModel):
    id: int
    description: str
    camera: Optional[str] = None
    mood: Optional[str] = None
    characters: List[str] = []
    caption: Optional[str] = None
    seed: Optional[int] = None

class Storyboard(BaseModel):
    title: str
    style: str
    panels: List[Panel]