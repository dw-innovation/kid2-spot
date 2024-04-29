from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


# Tag Combinations
class TagType(Enum):
    CORE = 'core'
    ATTR = 'attr'
    CORE_ATTR = 'core/attr'


# Tag Info
def remove_duplicate_tag_attributes(tag_attributes):
    processed_tag_attributes = []
    attribute_keys = []

    for tag_attribute in tag_attributes:
        if str(tag_attribute) in attribute_keys:
            continue
        else:
            # attribute_keys.append(str(tag_attribute))
            processed_tag_attributes.append(
                TagAttribute(descriptors=tag_attribute.descriptors, tags=tag_attribute.tags))
            # TagAttribute(key=tag_attribute.key, operator=tag_attribute.operator, value=tag_attribute.value))
    return processed_tag_attributes


class Tag(BaseModel, frozen=True):
    key: str = Field(description="Tag property key")
    operator: str = Field(description="Tag property operator")
    value: str = Field(description="Tag property value")


class TagAttribute(BaseModel, frozen=True):
    descriptors: List[str] = Field(description="List of text names")
    tags: List[Tag]  # MAYBE TAG INSTEAD???

    # name: str = Field(description='This will be filled out from descriptors')
    # key: str = Field(description="Tag property key")
    # operator: str = Field(description="Tag property operator")
    # value: str = Field(description="Tag property value")


class TagCombination(BaseModel):
    cluster_id: int = Field(description="Cluster Id")
    descriptors: List[str] = Field(description="List of text names")
    comb_type: TagType = Field(descripton="Tag type")
    tags: List[Tag] = Field(description="tags in the combination")
    tag_attributes: List[TagAttribute] = Field(description="List of tag attributes")


class TagAttributeExample(BaseModel):
    key: str
    examples: List[str]


# YAML Output
class Area(BaseModel):
    type: str
    value: str


class Property(BaseModel):
    name: str = Field(description='This will be filled out from descriptors')
    # key: str
    operator: Optional[
        str] = None  # Field(description='It is = for non-numerical properties, For other values, it can be =,<,>,~')
    value: Optional[str] = None


class Entity(BaseModel):
    id: int
    name: str
    type: str = Field(default='nwr')
    properties: Optional[List[Property]] = None


class Relation(BaseModel):
    name: str
    source: int
    target: int
    value: str


class Relations(BaseModel):
    relations: Optional[List[Relation]] = None
    type: str

    def update(self, **new_data):
        for field, value in new_data.items():
            setattr(self, field, value)


class LocPoint(BaseModel):
    area: Area
    entities: List[Entity]
    relations: Optional[Relations] = None


# Relative Spatial Terms
class RelSpatial(BaseModel):
    distance: str
    values: List[str]


##########################
# Data Model for Prompts #
##########################
class GeneratedPrompt(BaseModel):
    query: LocPoint
    prompt: str
    style: str
    persona: str

###################################################
# Data Model for Prompts and Sentence Generations #
###################################################
class GeneratedIMRSentence(BaseModel):
    query: LocPoint
    prompt: str
    style: str
    persona: str
    sentence: str

if __name__ == '__main__':
    area = Area(type='area', value='Berlin')
    assert area

    entities = [Entity(id=1, name='kiosk', type='nwr'),
                Entity(id=2, name='cafe', properties=[Property(name='name', key='name', operator='=', value='Luv')])]
    assert entities

    relations = Relations(type="", relations=[Relation(name='dist', source=0, target=1, value='10 km')])
    assert entities

    loc_point = LocPoint(area=area, entities=entities, relations=relations)
    assert loc_point

    loc_point = LocPoint(area=area, entities=entities)
    assert loc_point
