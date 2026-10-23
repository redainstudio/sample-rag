import json
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from app.api.deps import get_chat_service
from app.generation.service import GroundedChatService
from app.schemas.chat import ChatRequest, ChatResponse

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post(
    "/completions",
    response_model=ChatResponse,
    summary="Grounded chat completion with streaming or JSON response"
)
async def chat_completion(
    request: ChatRequest,
    service: GroundedChatService = Depends(get_chat_service)
):
    if not request.stream:
        try:
            return service.process_chat(request)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Chat generation failed: {e}"
            )

    # Streaming mode via Server-Sent Events (SSE)
    def event_stream():
        try:
            for event in service.process_chat_stream(request):
                # Format SSE protocol: event: {type}\ndata: {json_payload}\n\n
                data_str = (
                    json.dumps(event.data)
                    if isinstance(event.data, (dict, list))
                    else str(event.data)
                )
                yield f"event: {event.event.value}\ndata: {data_str}\n\n"
        except Exception as e:
            error_data = json.dumps({"error": str(e)})
            yield f"event: error\ndata: {error_data}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )
