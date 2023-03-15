from mlserver import MLModel
from mlserver.types import InferenceRequest, InferenceResponse
from mlserver.codecs import NumpyCodec
import numpy as np

class IsComplexConditional(MLModel):

    async def load(self) -> bool:
        self.ready = True
        return self.ready

    async def predict(self, payload: InferenceRequest) -> InferenceResponse:
        input = next(x for x in payload.inputs if x.name == "is_complex")
        is_complex = NumpyCodec.decode_input(input).astype(bool)[0]

        if is_complex:
            output_name = "is_complex_claim"
        else:
            output_name = "is_simple_claim"
        output_data = np.ones((1), dtype=bool)

        return InferenceResponse(
            id=payload.id,
            model_name=self.name,
            model_version=self.version,
            outputs=[
                NumpyCodec.encode_output(output_name, output_data)
            ]
        )