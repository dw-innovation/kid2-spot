from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Optional, Union


# Tag Combinations
class TagType(Enum):
    CORE = 'core'
    PROP = 'prop'
    CORE_PROP = 'core/prop'


# Tag Info
def remove_duplicate_tag_properties(tag_properties):
    processed_tag_properties = []
    # property_keys = []

    for tag_property in tag_properties:
        processed_tag_property = TagProperty(descriptors=tag_property.descriptors, tags=tag_property.tags)
        if processed_tag_property in processed_tag_properties:
            continue
        else:
            # property_keys.append(str(tag_property))
            processed_tag_properties.append(processed_tag_property)
            # TagProperty(key=tag_property.key, operator=tag_property.operator, value=tag_property.value))
    return processed_tag_properties


class Tag(BaseModel, frozen=True):
    key: str = Field(description="Tag property key")
    operator: str = Field(description="Tag property operator")
    value: str = Field(description="Tag property value")

    def to_dict(self):
        return {'key': self.key, 'operator': self.operator, 'value': self.value}


class TagProperty(BaseModel, frozen=True):
    descriptors: List[str] = Field(description="List of text names")
    tags: List[Tag]


class ColorBundle(BaseModel):
    descriptors: List[str] = Field(description="List of text names")
    color_values: List[str]

class TagCombination(BaseModel):
    cluster_id: int = Field(description="Cluster Id")
    is_area: bool = Field(description="Is this a area? True if it is, otherwise False")
    descriptors: List[str] = Field(description="List of text names")
    comb_type: TagType = Field(descripton="Tag type")
    tags: List[Tag] = Field(description="tags in the combination")
    tag_properties: List[TagProperty] = Field(description="List of tag properties")


class TagPropertyExample(BaseModel):
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


class Distance(BaseModel, frozen=True):
    magnitude: str
    metric: str


class Entity(BaseModel):
    id: int
    is_area: bool
    name: str
    type: str = Field(default='nwr')
    minPoints: Optional[int] = None
    maxDistance: Optional[Distance] = None
    properties: Optional[List[Property]] = None


class Relation(BaseModel):
    type: str
    source: int
    target: int
    value: Optional[Distance] = None


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

    def update_relations(self, new_relations_data):
        contain_relations = []
        if self.relations:
            contain_relations = [relation for relation in self.relations.relations if relation.type == "contains"]

        contain_relations.extend(new_relations_data.relations)
        self.relations.relations = contain_relations


# Relative Spatial Terms
class RelSpatial(BaseModel):
    distance: Distance
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
