import uuid
from sqlmodel import Session
from db import engine, create_db_and_tables
from models import Workspace, Domain, Datasource, DomainDatasource, DataColumn

# Define the workspaces as a list of dictionaries
workspaces = [
    {"workspace_id": 1, "workspace_name": "Sales Analytics"},
    {"workspace_id": 2, "workspace_name": "Marketing Insights"},
    {"workspace_id": 3, "workspace_name": "Product Development"},
]

domains = [
    {"workspace_id": 1, "domain_id": 101, "description": "Revenue Tracking", "purpose": "Monitor and analyze sales revenue trends."},
    {"workspace_id": 1, "domain_id": 102, "description": "Customer Segmentation", "purpose": "Identify and categorize customer demographics."},
    {"workspace_id": 2, "domain_id": 201, "description": "Campaign Performance", "purpose": "Evaluate the effectiveness of marketing campaigns."},
    {"workspace_id": 3, "domain_id": 301, "description": "Feature Prioritization", "purpose": "Determine which product features to prioritize based on user feedback."},
    {"workspace_id": 3, "domain_id": 302, "description": "Prototype Testing", "purpose": "Test prototypes for usability and performance."},
]

domain_datasources = [
    {"domain_id": 101, "datasource_id": 1001, "description": "Sales Transactions Table","datasource":True},
    {"domain_id": 101, "datasource_id": 1002, "description": "Customer Feedback Table","datasource":True},
    {"domain_id": 102, "datasource_id": 1003, "description": "Demographics Data Table","datasource":True},
    {"domain_id": 201, "datasource_id": 1004, "description": "Campaign Metrics Table","datasource":True},
    {"domain_id": 201, "datasource_id": 1005, "description": "Social Media Insights Table","datasource":True},
    {"domain_id": 301, "datasource_id": 1006, "description": "Feature Requests Table","datasource":True},
    {"domain_id": 302, "datasource_id": 1007, "description": "Prototype Test Results Table","datasource":True},
]

columns = [
    {
        "datasource_id": 1001, "column_id": 1, "column_name": "transaction_amount",
        "column_type": "number",
        "column_description": "The total monetary value of individual transactions processed.",
        "column_stats": {"min": 0, "max": 10000, "mode": 500, "median": 1000, "avg": 1200},
        "column_indexed": True, "column_partitioned": False
    },
    {
        "datasource_id": 1001, "column_id": 2, "column_name": "transaction_category",
        "column_type": "string",
        "column_description": "The category defining the type or nature of transactions conducted.",
        "column_stats": {"unique_values": ["Retail", "Wholesale", "Online", "In-store", "Other"]},
        "column_indexed": False, "column_partitioned": True
    },
    {
        "datasource_id": 1001, "column_id": 3, "column_name": "transaction_count",
        "column_type": "number",
        "column_description": "The count of transactions performed by customers in the system.",
        "column_stats": {"min": 1, "max": 500, "mode": 50, "median": 100, "avg": 150},
        "column_indexed": True, "column_partitioned": False
    },
    {
        "datasource_id": 1001, "column_id": 4, "column_name": "payment_method",
        "column_type": "string",
        "column_description": "The method used by customers to pay for transactions (e.g., cash).",
        "column_stats": {"unique_values": ["Credit Card", "Debit Card", "Cash", "Digital Wallet", "Other"]},
        "column_indexed": True, "column_partitioned": True
    },
    {
        "datasource_id": 1001, "column_id": 5, "column_name": "customer_age",
        "column_type": "number",
        "column_description": "The age of the customer performing transactions, measured in years.",
        "column_stats": {"min": 18, "max": 75, "mode": 30, "median": 35, "avg": 40},
        "column_indexed": False, "column_partitioned": False
    },
    {
        "datasource_id": 1001, "column_id": 6, "column_name": "customer_region",
        "column_type": "string",
        "column_description": "The geographical region where the customer resides or operates.",
        "column_stats": {"unique_values": ["North", "South", "East", "West", "Central"]},
        "column_indexed": False, "column_partitioned": True
    },
    {
        "datasource_id": 1001, "column_id": 7, "column_name": "discount_percentage",
        "column_type": "number",
        "column_description": "The percentage discount applied to transactions for promotions or offers.",
        "column_stats": {"min": 0, "max": 50, "mode": 10, "median": 15, "avg": 12},
        "column_indexed": True, "column_partitioned": False
    },
    {
        "datasource_id": 1001, "column_id": 8, "column_name": "product_category",
        "column_type": "string",
        "column_description": "The category of products purchased in transactions (e.g., Electronics).",
        "column_stats": {"unique_values": ["Electronics", "Clothing", "Grocery", "Furniture", "Toys"]},
        "column_indexed": False, "column_partitioned": False
    },
    {
        "datasource_id": 1001, "column_id": 9, "column_name": "units_sold",
        "column_type": "number",
        "column_description": "The total number of product units sold during transactions.",
        "column_stats": {"min": 1, "max": 1000, "mode": 100, "median": 200, "avg": 250},
        "column_indexed": True, "column_partitioned": True
    },
    {
        "datasource_id": 1001, "column_id": 10, "column_name": "sales_representative",
        "column_type": "string",
        "column_description": "The sales representative responsible for handling the transaction or sale.",
        "column_stats": {"unique_values": ["John", "Jane", "Smith", "Doe", "Other"]},
        "column_indexed": True, "column_partitioned": False
    }
]

# Add meaningful data for other datasources
columns.extend([
    {
        "datasource_id": 1002, "column_id": 1, "column_name": "feedback_score",
        "column_type": "number",
        "column_description": "Numerical score representing customer feedback on a scale of 1 to 10.",
        "column_stats": {"min": 1, "max": 10, "mode": 8, "median": 7, "avg": 7.5},
        "column_indexed": True, "column_partitioned": False
    },
    {
        "datasource_id": 1002, "column_id": 2, "column_name": "feedback_comment",
        "column_type": "string",
        "column_description": "Textual comments provided by customers to express their feedback or opinion.",
        "column_stats": {"unique_values": ["Excellent", "Good", "Average", "Poor", "Bad"]},
        "column_indexed": False, "column_partitioned": False
    },
    {
        "datasource_id": 1003, "column_id": 1, "column_name": "age_group",
        "column_type": "string",
        "column_description": "Categorical age ranges representing the demographic of users or customers.",
        "column_stats": {"unique_values": ["18-25", "26-35", "36-45", "46-60", "60+"]},
        "column_indexed": True, "column_partitioned": False
    },
    {
        "datasource_id": 1003, "column_id": 2, "column_name": "income_bracket",
        "column_type": "number",
        "column_description": "The income range of individuals, represented in monetary terms per annum.",
        "column_stats": {"min": 10000, "max": 200000, "mode": 75000, "median": 85000, "avg": 90000},
        "column_indexed": True, "column_partitioned": True
    }
])

columns.extend([
    # Datasource 1004: Campaign Metrics Table
    {
        "datasource_id": 1004, "column_id": 1, "column_name": "campaign_id",
        "column_type": "number",
        "column_description": "Unique identifier assigned to each marketing campaign.",
        "column_stats": {"min": 1, "max": 5000, "mode": 100, "median": 250, "avg": 300},
        "column_indexed": True, "column_partitioned": False
    },
    {
        "datasource_id": 1004, "column_id": 2, "column_name": "impressions",
        "column_type": "number",
        "column_description": "The total number of times the campaign was displayed to users.",
        "column_stats": {"min": 100, "max": 1000000, "mode": 50000, "median": 100000, "avg": 250000},
        "column_indexed": True, "column_partitioned": False
    },
    {
        "datasource_id": 1004, "column_id": 3, "column_name": "clicks",
        "column_type": "number",
        "column_description": "The total number of clicks received by the campaign's ads.",
        "column_stats": {"min": 10, "max": 50000, "mode": 1000, "median": 5000, "avg": 10000},
        "column_indexed": True, "column_partitioned": True
    },
    {
        "datasource_id": 1004, "column_id": 4, "column_name": "conversion_rate",
        "column_type": "number",
        "column_description": "Percentage of users who completed a desired action after clicking.",
        "column_stats": {"min": 0.1, "max": 20, "mode": 2.5, "median": 5, "avg": 7},
        "column_indexed": False, "column_partitioned": False
    },
    {
        "datasource_id": 1004, "column_id": 5, "column_name": "spend",
        "column_type": "number",
        "column_description": "Total amount of money spent on the marketing campaign.",
        "column_stats": {"min": 100, "max": 100000, "mode": 5000, "median": 10000, "avg": 25000},
        "column_indexed": True, "column_partitioned": True
    },


    # Datasource 1005: Social Media Insights Table

    {
        "datasource_id": 1005, "column_id": 1, "column_name": "platform",
        "column_type": "string",
        "column_description": "The social media platform where the campaign is being run.",
        "column_stats": {"unique_values": ["Facebook", "Instagram", "Twitter", "LinkedIn", "TikTok"]},
        "column_indexed": False, "column_partitioned": True
    },
    {
        "datasource_id": 1005, "column_id": 2, "column_name": "followers",
        "column_type": "number",
        "column_description": "The total number of followers on the specified social platform.",
        "column_stats": {"min": 1000, "max": 10000000, "mode": 50000, "median": 100000, "avg": 250000},
        "column_indexed": True, "column_partitioned": False
    },
    {
        "datasource_id": 1005, "column_id": 3, "column_name": "engagement_rate",
        "column_type": "number",
        "column_description": "The percentage of followers interacting with posts on the platform.",
        "column_stats": {"min": 0.1, "max": 15, "mode": 2.5, "median": 5, "avg": 6},
        "column_indexed": False, "column_partitioned": False
    },
    {
        "datasource_id": 1005, "column_id": 4, "column_name": "post_frequency",
        "column_type": "number",
        "column_description": "The average number of posts made weekly on the platform.",
        "column_stats": {"min": 1, "max": 50, "mode": 5, "median": 10, "avg": 15},
        "column_indexed": False, "column_partitioned": True
    },
    {
        "datasource_id": 1005, "column_id": 5, "column_name": "top_post",
        "column_type": "string",
        "column_description": "The type of post that performs the best in terms of engagement.",
        "column_stats": {"unique_values": ["Image", "Video", "Text Post", "Story", "Reel"]},
        "column_indexed": True, "column_partitioned": False
    }
,

    # Datasource 1006: Feature Requests Table
    {
        "datasource_id": 1006, "column_id": 1, "column_name": "request_id",
        "column_type": "number",
        "column_description": "A unique identifier for each feature request submitted.",
        "column_stats": {"min": 1, "max": 10000, "mode": 500, "median": 1500, "avg": 2500},
        "column_indexed": True, "column_partitioned": False
    },
    {
        "datasource_id": 1006, "column_id": 2, "column_name": "feature_name",
        "column_type": "string",
        "column_description": "The name of the feature being requested or suggested by users.",
        "column_stats": {"unique_values": ["Search", "Notifications", "Analytics", "Profile Management", "Integration"]},
        "column_indexed": False, "column_partitioned": True
    },
    {
        "datasource_id": 1006, "column_id": 3, "column_name": "votes",
        "column_type": "number",
        "column_description": "The total number of votes a feature request has received from users.",
        "column_stats": {"min": 0, "max": 10000, "mode": 100, "median": 500, "avg": 1500},
        "column_indexed": True, "column_partitioned": False
    },
    {
        "datasource_id": 1006, "column_id": 4, "column_name": "priority",
        "column_type": "string",
        "column_description": "The assigned priority level for implementing the requested feature.",
        "column_stats": {"unique_values": ["Low", "Medium", "High", "Critical"]},
        "column_indexed": False, "column_partitioned": False
    }
,

    {
        "datasource_id": 1007, "column_id": 1, "column_name": "test_case_id",
        "column_type": "number",
        "column_description": "A unique identifier for each test case executed during testing.",
        "column_stats": {"min": 1, "max": 2000, "mode": 100, "median": 500, "avg": 750},
        "column_indexed": True, "column_partitioned": False
    },
    {
        "datasource_id": 1007, "column_id": 2, "column_name": "test_result",
        "column_type": "string",
        "column_description": "The outcome of the test case execution (e.g., Pass or Fail).",
        "column_stats": {"unique_values": ["Pass", "Fail", "Blocked", "Skipped"]},
        "column_indexed": False, "column_partitioned": True
    },
    {
        "datasource_id": 1007, "column_id": 3, "column_name": "execution_time",
        "column_type": "number",
        "column_description": "The time taken to execute the test case, measured in milliseconds.",
        "column_stats": {"min": 10, "max": 5000, "mode": 200, "median": 500, "avg": 750},
        "column_indexed": True, "column_partitioned": False
    },
    {
        "datasource_id": 1007, "column_id": 4, "column_name": "tester_name",
        "column_type": "string",
        "column_description": "The name of the tester responsible for executing the test case.",
        "column_stats": {"unique_values": ["Alice", "Bob", "Charlie", "Dave", "Eve"]},
        "column_indexed": False, "column_partitioned": True
    }


])

def add_meaningful_data():
    # Ensure tables exist
    create_db_and_tables()

    # Mappings: old numeric ID -> newly generated UUID
    workspace_uuid_map = {}
    domain_uuid_map = {}
    datasource_uuid_map = {}

    with Session(engine) as session:
        # 1) Insert Workspaces, generating UUIDs
        for w in workspaces:
            w_uuid = uuid.uuid4()
            workspace_uuid_map[w["workspace_id"]] = w_uuid

            workspace = Workspace(
                id=w_uuid,
                name=w["workspace_name"]  # "Sales Analytics", etc.
            )
            session.add(workspace)

        session.commit()

        # 2) Insert Domains, linking to the correct Workspace via the map
        for d in domains:
            d_uuid = uuid.uuid4()
            domain_uuid_map[d["domain_id"]] = d_uuid

            mapped_workspace_id = workspace_uuid_map[d["workspace_id"]]

            # “description” from input becomes the Domain’s `name` and `description`.
            # You might choose to store them differently if you prefer.
            domain = Domain(
                id=d_uuid,
                name=d["description"],          # e.g. "Revenue Tracking"
                description=d["description"],   # duplicating the same text, or store something else
                purpose=d["purpose"],           # e.g. "Monitor and analyze sales revenue trends."
                workspace_id=mapped_workspace_id,
            )
            session.add(domain)

        session.commit()

        ds_id_description_map = {}
        for ds in domain_datasources:
            ds_id = ds["datasource_id"]
            # If a datasource_id is repeated with different descriptions, you may need extra logic
            # but in this example, each ds_id has exactly one "description"
            ds_id_description_map[ds_id] = ds["description"]

        # Convert them to real Datasource rows
        for ds_id, ds_description in ds_id_description_map.items():
            ds_uuid = uuid.uuid4()
            datasource_uuid_map[ds_id] = ds_uuid

            # Use the stored description
            datasource = Datasource(
                id=ds_uuid,
                description=ds_description
            )
            session.add(datasource)
        session.commit()

        # 4) Insert DomainDatasource association rows
        for dd in domain_datasources:
            mapped_domain_id = domain_uuid_map[dd["domain_id"]]
            mapped_datasource_id = datasource_uuid_map[dd["datasource_id"]]

            domain_datasource = DomainDatasource(
                id=uuid.uuid4(),
                domain_id=mapped_domain_id,
                datasource_id=mapped_datasource_id,
                # There is no "description" field in DomainDatasource, so skip it here
            )
            session.add(domain_datasource)
        session.commit()


        # 5) Insert DataColumns, referencing each mapped Datasource’s UUID.
        #    Note: The DataColumn model only has `column_dec` and `datasource_id`.
        #    We combine `column_name` + `column_description` into `column_dec`.
        for col in columns:
            mapped_datasource_id = datasource_uuid_map[col["datasource_id"]]

            # Combine “column_name” + “column_description” into the single `column_dec` field.
            combined_desc = f"{col['column_name']}: {col['column_description']}"
            data_column = DataColumn(
                id=uuid.uuid4(),
                datasource_id=mapped_datasource_id,
                column_dec=combined_desc,
            )
            session.add(data_column)

        session.commit()

        print("Meaningful data inserted successfully!")
        # Optionally, you can print or debug the UUID mappings here.
        # print("Workspace UUIDs:", workspace_uuid_map)
        # print("Domain UUIDs:", domain_uuid_map)
        # print("Datasource UUIDs:", datasource_uuid_map)


if __name__ == "__main__":
    add_meaningful_data()
