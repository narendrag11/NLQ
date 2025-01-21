from datetime import datetime, timezone, timedelta
import re
from typing import Optional, List, Union, Literal, Any
from uuid import UUID, uuid4
from enum import Enum
from pydantic import Json, Field, validator, ValidationInfo, field_validator, model_validator
from sqlalchemy import Column,  UniqueConstraint
from sqlmodel import Field, Relationship, SQLModel, JSON
from sqlalchemy.dialects.postgresql import JSONB
from pgvector.sqlalchemy import Vector


class Workspace(SQLModel, table=True):
    # __table_args__ = {"public": "workspace"}
    __tablename__="workspace"
   
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    name: str

    # One Workspace can have multiple Domains
    domains: List["Domain"] = Relationship(back_populates="workspace")


class Domain(SQLModel, table=True):

    __tablename__ = "domain"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    name: str
    description: Optional[str] = Field(default=None)
    purpose: Optional[str] = Field(default=None)
    workspace_id: UUID = Field(foreign_key="workspace.id")

    # Relationship back to Workspace
    workspace: Optional[Workspace] = Relationship(back_populates="domains")

    # A Domain can be linked to multiple Datasources through the association table
    domain_datasources: List["DomainDatasource"] = Relationship(back_populates="domain")


class Datasource(SQLModel, table=True):
    # __table_args__ = {"public": "datasource"}
    
    __tablename__ = "datasource"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    description: Optional[str] = Field(default=None)

    # A Datasource can be linked to multiple Domains through the association table
    domain_datasources: List["DomainDatasource"] = Relationship(back_populates="datasource")

    # A Datasource can have multiple DataColumns
    data_columns: List["DataColumn"] = Relationship(back_populates="datasource")


class DomainDatasource(SQLModel, table=True):
    """
    Association (mapping) table between Domain and Datasource
    """
    __tablename__ = "domain_datasource"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    domain_id: UUID = Field(foreign_key="domain.id")
    datasource_id: UUID = Field(foreign_key="datasource.id")

    # Relationships
    domain: "Domain" = Relationship(back_populates="domain_datasources")
    datasource: "Datasource" = Relationship(back_populates="domain_datasources")


class DataColumn(SQLModel, table=True):
    __tablename__ = "data_column"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    column_dec: str
    datasource_id: UUID = Field(foreign_key="datasource.id")
    # Relationship back to Datasource
    datasource: Optional[Datasource] = Relationship(back_populates="data_columns")


class DomainEmbedding(SQLModel, table=True):
    __table_args__ = {"schema": "embedding"}
    id : UUID = Field(default_factory=uuid4, primary_key=True)
    domain_id: UUID = Field(foreign_key="domain.id")
    embedding: Any = Field(sa_column=Column(Vector(768)))
    embedding_type:str # it will be decription or purpose 


class ColumnEmbedding(SQLModel, table=True):
    __table_args__ = {"schema": "embedding"}
    id : UUID = Field(default_factory=uuid4, primary_key=True)
    column_id: UUID = Field(foreign_key="data_column.id")
    embedding: Any = Field(sa_column=Column(Vector(768)))

    
class TableEmbedding(SQLModel, table=True):
    __table_args__ = {"schema": "embedding"}
    id : UUID = Field(default_factory=uuid4, primary_key=True)
    table_id: UUID = Field(foreign_key="datasource.id")
    embedding: Any = Field(sa_column=Column(Vector(768)))

    

# # Domain, Column, and table embedding for vector dimension 5
# class DomainEmbedding_5(SQLModel, table=True):
#     __table_args__ = {"schema": "embedding"}
#     id : UUID = Field(default_factory=uuid4, primary_key=True)
#     domain_id: UUID = Field(foreign_key="domain.id")
#     embedding: Any = Field(sa_column=Column(Vector(5)))
    
# class ColumnEmbedding_5(SQLModel, table=True):
#     __table_args__ = {"schema": "embedding"}
#     id : UUID = Field(default_factory=uuid4, primary_key=True)
#     column_id: UUID = Field(foreign_key="column.id")
#     embedding: Any = Field(sa_column=Column(Vector(5)))

# class TableEmbedding_5(SQLModel, table=True):
#     __table_args__ = {"schema": "embedding"}
#     id : UUID = Field(default_factory=uuid4, primary_key=True)
#     table_id: UUID = Field(foreign_key="datasource.id")
#     embedding: Any = Field(sa_column=Column(Vector(5)))