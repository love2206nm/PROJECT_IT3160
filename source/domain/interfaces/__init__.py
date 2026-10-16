"""
Domain Interfaces / Protocols package.
Exports standard domain abstractions for Clean Architecture.
"""

from source.domain.interfaces.data_pipeline import (
    IDataLoader,
    IFeatureTransformer,
    IMetadataBuilder,
)
from source.domain.interfaces.onet_repository import IOnetRepository
from source.domain.interfaces.parser import (
    IDocumentParser,
    IProfileExtractor,
)
from source.domain.interfaces.ml_models import (
    IClusteringEngine,
    IRiskPredictor,
)
from source.domain.interfaces.rag import (
    IEvidenceRetriever,
)
from source.domain.interfaces.copilot import ICopilotWorkflow

__all__ = [
    # Data Pipeline protocols
    "IDataLoader",
    "IFeatureTransformer",
    "IMetadataBuilder",
    # Repository protocol
    "IOnetRepository",
    # Parser protocols
    "IDocumentParser",
    "IProfileExtractor",
    # Machine Learning protocols
    "IClusteringEngine",
    "IRiskPredictor",
    # RAG protocol
    "IEvidenceRetriever",
    # Copilot Workflow protocol
    "ICopilotWorkflow",
]
#@runtime_checkable là để giúp các lớp kế thừa từ các lớp đã kế thừa protocol theo đúng khuôn
#mẫu đã đặt ra từ những lớp kế thừa protocol 
#protocol sẽ giúp các lớp tạo ra như 1 cái vỏ bọc nghĩa là nếu bất kỳ thg nào muốn truy cập vào 
#db thì cũng cần có cái khuôn giống lớp đấy
