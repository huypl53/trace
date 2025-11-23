"""Type definitions for table line extraction pipeline."""

from typing import Literal
from pydantic import BaseModel, model_validator


class Line(BaseModel):
    """Represents a table line segment (horizontal or vertical)."""

    x1: float
    y1: float
    x2: float
    y2: float
    direction: Literal['vertical', 'horizontal']
    visible: bool

    @model_validator(mode='after')
    def validate_line_direction(self):
        """Validate that line coordinates match the direction."""
        if self.direction == 'vertical':
            # For vertical lines, x1 must equal x2
            if abs(self.x1 - self.x2) > 1e-6:  # Allow small float precision errors
                raise ValueError(f"Vertical line must have x1 == x2, got x1={self.x1}, x2={self.x2}")
        elif self.direction == 'horizontal':
            # For horizontal lines, y1 must equal y2
            if abs(self.y1 - self.y2) > 1e-6:  # Allow small float precision errors
                raise ValueError(f"Horizontal line must have y1 == y2, got y1={self.y1}, y2={self.y2}")
        return self

    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            'x1': self.x1,
            'y1': self.y1,
            'x2': self.x2,
            'y2': self.y2,
            'direction': self.direction,
            'visible': self.visible,
        }
