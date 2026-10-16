"""
Constants and mappings for O*NET Data Engineering and Preprocessing.
"""

from typing import Dict, List, Any

# 23 O*NET Major Groups (2-digit SOC prefixes)
ONET_MAJOR_GROUPS: Dict[str, str] = {
    "11": "Management Occupations",
    "13": "Business and Financial Operations Occupations",
    "15": "Computer and Mathematical Occupations",
    "17": "Architecture and Engineering Occupations",
    "19": "Life, Physical, and Social Science Occupations",
    "21": "Community and Social Service Occupations",
    "23": "Legal Occupations",
    "25": "Educational Instruction and Library Occupations",
    "27": "Arts, Design, Entertainment, Sports, and Media Occupations",
    "29": "Healthcare Practitioners and Technical Occupations",
    "31": "Healthcare Support Occupations",
    "33": "Protective Service Occupations",
    "35": "Food Preparation and Serving Related Occupations",
    "37": "Building and Grounds Cleaning and Maintenance Occupations",
    "39": "Personal Care and Service Occupations",
    "41": "Sales and Related Occupations",
    "43": "Office and Administrative Support Occupations",
    "45": "Farming, Fishing, and Forestry Occupations",
    "47": "Construction and Extraction Occupations",
    "49": "Installation, Maintenance, and Repair Occupations",
    "51": "Production Occupations",
    "53": "Transportation and Material Moving Occupations",
    "55": "Military Specific Occupations",
}

# 20 Software Taxonomy Categories with pattern keywords for matching Element Names in software_skills.csv
SOFTWARE_TAXONOMY_MAP: Dict[str, List[str]] = {
    "tech_cat_data_query_sql": [
        "data base management",
        "data base user interface and query",
        "database reporting",
        "object oriented data base",
    ],
    "tech_cat_development_env": [
        "development environment",
        "integrated development environment",
    ],
    "tech_cat_oop_programming": [
        "object or component oriented development",
        "compiler and decompiler",
        "program testing",
    ],
    "tech_cat_web_platform": [
        "web platform development",
        "internet directory services",
        "web page creation and editing",
    ],
    "tech_cat_cloud_os_devops": [
        "operating system",
        "configuration management",
        "platform interconnectivity",
    ],
    "tech_cat_analytics_scientific": [
        "analytical or scientific",
        "pattern recognition",
        "expert system",
    ],
    "tech_cat_cad_engineering": [
        "computer aided design cad",
        "computer aided manufacturing cam",
        "industrial control",
    ],
    "tech_cat_enterprise_erp": [
        "enterprise resource planning erp",
        "materials requirements planning mrp",
        "supply chain",
    ],
    "tech_cat_crm_sales": [
        "customer relationship management crm",
        "sales and marketing",
        "point of sale",
    ],
    "tech_cat_bi_reporting": [
        "business intelligence and data analysis",
        "data mining",
    ],
    "tech_cat_multimedia_creative": [
        "graphics or photo imaging",
        "video creation and editing",
        "audio editing",
        "desktop publishing",
    ],
    "tech_cat_security_network": [
        "network monitoring",
        "network security",
        "security and protection",
        "network directory",
    ],
    "tech_cat_project_management": [
        "project management",
        "time accounting",
        "calendar and scheduling",
    ],
    "tech_cat_document_management": [
        "document management",
        "information retrieval or search",
        "optical character recognition",
    ],
    "tech_cat_gis_spatial": [
        "geographic information system",
        "map creation",
    ],
    "tech_cat_medical_clinical": [
        "medical",
        "electronic health record",
        "clinical information",
    ],
    "tech_cat_accounting_finance": [
        "financial analysis",
        "accounting",
        "tax preparation",
        "billing",
    ],
    "tech_cat_office_spreadsheets": [
        "spreadsheet",
        "word processing",
        "presentation",
    ],
    "tech_cat_communication_collab": [
        "electronic mail",
        "video conference",
        "instant messaging",
    ],
    "tech_cat_mobile_embedded": [
        "mobile application development",
        "microcontroller programming",
    ],
}

# Unnecessary statistical survey columns to drop across O*NET files
SURVEY_NOISE_COLUMNS: List[str] = [
    "N",
    "Standard Error",
    "Lower 95% Limit",
    "Upper 95% Limit",
    "Suppress",
    "Date",
    "Domain Source",
    "Category",
]

# Benchmark Risk Profiles for 33 SOCs missing in threat survey (45-xxxx and 55-xxxx)
UNMATCHED_SOC_BENCHMARKS: Dict[str, Dict[str, Any]] = {
    "45": {
        "threat_score": 35.0,
        "risk_level": "Low",
        "sub_task_automation": 30.0,
        "sub_tool_maturity": 25.0,
        "sub_agentic_exposure": 15.0,
        "defense_skills": [
            "Physical Dexterity",
            "Environmental Adaptation",
            "Hands-on Coordination",
        ],
        "insight": "Manual agricultural operations, biological intuition, and dynamic environmental field navigation possess strong physical defense against digital AI automation.",
    },
    "55": {
        "threat_score": 15.0,
        "risk_level": "Low",
        "sub_task_automation": 15.0,
        "sub_tool_maturity": 20.0,
        "sub_agentic_exposure": 10.0,
        "defense_skills": [
            "Crisis Decision Making",
            "Tactical Leadership",
            "Physical Resilience",
        ],
        "insight": "Tactical combat operations, high-stakes military leadership, and physical resilience exhibit near-zero automation threat in defense doctrines.",
    },
}
