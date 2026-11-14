"""Verify request contracts and failure handling without network calls."""
import json
import unittest
from unittest.mock import patch

import httpx
from pydantic import BaseModel, Field
import provider


class Result(BaseModel):
    answer: str
    details: list[str] = Field(default_factory=list)


class ProviderTests(unittest.IsolatedAsyncioTestCase):
    def client(self, handler):
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        return patch.object(provider.httpx, 'AsyncClient', return_value=client)

    async def test_schema_image_and_bearer_auth(self):
        def handler(request):
            self.assertEqual(request.headers['Authorization'], 'Bearer test-key')
            payload=json.loads(request.content)
            self.assertEqual(payload['messages'][1]['content'][0]['type'],'image_url')
            schema=payload['response_format']['json_schema']['schema']
            self.assertEqual(schema['required'],['answer','details'])
            self.assertFalse(schema['additionalProperties'])
            return httpx.Response(200,json={'choices':[{'finish_reason':'stop','message':{'content':'{"answer":"red","details":[]}'}}]})
        with patch.dict('os.environ',{'CRUSOE_API_KEY':'test-key'}),self.client(handler):
            result=await provider.structured('Describe the image.',[{'type':'image_url','image_url':{'url':'data:image/jpeg;base64,AA=='}}],Result)
        self.assertEqual(result.answer,'red')

    async def test_provider_error_does_not_echo_body_or_secret(self):
        with patch.dict('os.environ',{'CRUSOE_API_KEY':'test-secret'}),self.client(lambda request:httpx.Response(401,text='test-secret')):
            with self.assertRaises(provider.InferenceError) as caught:
                await provider.complete([{'role':'user','content':'Hello'}])
        self.assertIn('401',str(caught.exception))
        self.assertNotIn('test-secret',str(caught.exception))

    async def test_truncated_and_invalid_structured_results_rejected(self):
        for finish,content in [('length','{"answer":"red","details":[]}'),('stop','not json')]:
            with self.subTest(finish=finish),patch.dict('os.environ',{'CRUSOE_API_KEY':'test-key'}),self.client(lambda request:httpx.Response(200,json={'choices':[{'finish_reason':finish,'message':{'content':content}}]})):
                with self.assertRaises(provider.InferenceError):
                    await provider.structured('Describe.','Input',Result)

    async def test_missing_key_fails_before_network(self):
        with patch.dict('os.environ',{'CRUSOE_API_KEY':''}),patch.object(provider.httpx,'AsyncClient') as client:
            with self.assertRaises(provider.InferenceError):
                await provider.complete([])
            client.assert_not_called()


class FallbackTests(unittest.IsolatedAsyncioTestCase):
    """A failed Crusoe call is retried on the configured fallback model."""

    def client(self, handler):
        """Route every request to a handler through one reusable mock client.

        Args:
            handler: Callable returning a response for a request.

        Returns:
            A patch that makes provider use the mock client.
        """
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        return patch.object(provider.httpx, 'AsyncClient', return_value=client)

    async def test_failed_primary_retries_on_fallback_model(self):
        """A 503 from the primary model is answered by the fallback model."""
        models = []

        def handler(request):
            model = json.loads(request.content)['model']
            models.append(model)
            if model == provider.MODEL:
                return httpx.Response(503)
            return httpx.Response(200, json={'choices': [{'finish_reason': 'stop', 'message': {'content': 'Hi'}}]})

        with patch.dict('os.environ', {'CRUSOE_API_KEY': 'k'}), self.client(handler):
            self.assertEqual(await provider.complete([{'role': 'user', 'content': 'Hello'}]), 'Hi')
        self.assertEqual(models, [provider.MODEL, provider.CONFIG.inference.fallback_models[0]])

    async def test_vision_call_uses_vision_fallbacks(self):
        """An image request falls back to the configured vision fallback model."""
        models = []

        def handler(request):
            model = json.loads(request.content)['model']
            models.append(model)
            if len(models) == 1:
                return httpx.Response(500)
            return httpx.Response(200, json={'choices': [{'finish_reason': 'stop', 'message': {'content': '{"answer":"red","details":[]}'}}]})

        with patch.dict('os.environ', {'CRUSOE_API_KEY': 'k'}), self.client(handler):
            result = await provider.structured('Describe.', 'Input', Result, model=provider.VISION_MODEL)
        self.assertEqual(result.answer, 'red')
        self.assertEqual(models[1], provider.CONFIG.inference.vision_fallback_models[0])

    async def test_invalid_structured_answer_retries_on_fallback(self):
        """Data that fails validation from the primary is retried on the fallback."""
        contents = iter(['not json', '{"answer":"blue","details":[]}'])

        def handler(request):
            return httpx.Response(200, json={'choices': [{'finish_reason': 'stop', 'message': {'content': next(contents)}}]})

        with patch.dict('os.environ', {'CRUSOE_API_KEY': 'k'}), self.client(handler):
            self.assertEqual((await provider.structured('Describe.', 'Input', Result)).answer, 'blue')

    async def test_all_models_failing_raises(self):
        """When every model fails, the caller still gets a safe InferenceError."""
        with patch.dict('os.environ', {'CRUSOE_API_KEY': 'k'}), self.client(lambda request: httpx.Response(503)):
            with self.assertRaises(provider.InferenceError):
                await provider.complete([{'role': 'user', 'content': 'Hello'}])

    async def test_fallback_is_reported_to_the_flow_view(self):
        """The flow view hears which model replaced the failed one."""
        from library_claim import telemetry
        events = []

        def handler(request):
            if json.loads(request.content)['model'] == provider.MODEL:
                return httpx.Response(503)
            return httpx.Response(200, json={'choices': [{'finish_reason': 'stop', 'message': {'content': 'Hi'}}]})

        with patch.dict('os.environ', {'CRUSOE_API_KEY': 'k'}), self.client(handler), telemetry.listening(events.append):
            await provider.complete([{'role': 'user', 'content': 'Hello'}])
        self.assertEqual(events[0]['stage'], 'fallback')
        self.assertIn(provider.CONFIG.inference.fallback_models[0], events[0]['detail'])
