from typing import List, Optional

from pydantic import BaseModel, Field


class Area(BaseModel):
    type: str
    value: str


class Property(BaseModel):
    name: str
    operator: str = Field(description='It is = for non-numerical properties, For other values, it can be =,<,>')
    value: str


class Entity(BaseModel):
    id: int
    name: str
    type: str = Field(default='nwr')
    properties: Optional[List[Property]]


class Relation(BaseModel):
    name: str
    source: int
    target: int
    value: str


class Relations(BaseModel):
    relations: List[Relation]


class LocPoint(BaseModel):
    area: Area
    entities: List[Entity]
    relations: Optional[List[Relation]]


if __name__ == '__main__':
    area = Area(type='area', value='Berlin')
    assert area

    entities = [Entity(id=1, name='kiosk'),
                Entity(id=2, name='cafe', properties=[Property(name='name', operator='=', value='Luv')])]
    assert entities

    relations = [Relation(name='dist', source=0, target=1, value='10 km')]
    assert entities

    loc_point = LocPoint(area=area, entities=entities, relations=relations)
    assert loc_point

    loc_point = LocPoint(area=area, entities=entities)
    assert loc_point
