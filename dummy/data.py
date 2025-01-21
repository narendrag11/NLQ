from uuid import uuid4
from sqlmodel import Session
from models import Workspace, Domain, Datasource, DomainDatasource, DataColumn
from db import engine  # Assumes you have a `db.py` that creates the `engine`

def populate_dummy_data():
    with Session(engine) as session:
        # 1. Create Workspaces
        workspaces = []
        for i in range(3):
            workspace = Workspace(name=f"Workspace {i + 1}")
            session.add(workspace)
            workspaces.append(workspace)

        session.commit()  # Commit to get IDs for Workspaces
        print("Workspaces created:", [workspace.id for workspace in workspaces])

        # 2. Create Domains linked to Workspaces
        domains = []
        for i, workspace in enumerate(workspaces):
            for j in range(2):  # Each workspace will have 2 domains
                domain = Domain(
                    name=f"Domain {i + 1}-{j + 1}",
                    description=f"Description for Domain {i + 1}-{j + 1}",
                    purpose=f"Purpose of Domain {i + 1}-{j + 1}",
                    workspace_id=workspace.id,
                )
                session.add(domain)
                domains.append(domain)

        session.commit()  # Commit to get IDs for Domains
        print("Domains created:", [domain.id for domain in domains])

        # 3. Create Datasources
        datasources = []
        for i in range(5):
            datasource = Datasource(description=f"Datasource {i + 1}")
            session.add(datasource)
            datasources.append(datasource)

        session.commit()  # Commit to get IDs for Datasources
        print("Datasources created:", [datasource.id for datasource in datasources])

        # 4. Create DomainDatasource (Association Table)
        domain_datasources = []
        for domain in domains:
            for i in range(2):  # Each domain will be linked to 2 datasources
                datasource = datasources[(domain.id.int + i) % len(datasources)]  # Randomly assign datasources
                domain_datasource = DomainDatasource(
                    domain_id=domain.id,
                    datasource_id=datasource.id,
                )
                session.add(domain_datasource)
                domain_datasources.append(domain_datasource)

        session.commit()
        print("DomainDatasource associations created:", [(dd.domain_id, dd.datasource_id) for dd in domain_datasources])

        # 5. Create DataColumns linked to Datasources
        data_columns = []
        for datasource in datasources:
            for i in range(3):  # Each datasource will have 3 data columns
                data_column = DataColumn(
                    column_dec=f"Column Description {datasource.id}-{i + 1}",
                    datasource_id=datasource.id,
                )
                session.add(data_column)
                data_columns.append(data_column)

        session.commit()
        print("DataColumns created:", [data_column.id for data_column in data_columns])

    print("Dummy data population complete!")


if __name__ == "__main__":
    populate_dummy_data()
