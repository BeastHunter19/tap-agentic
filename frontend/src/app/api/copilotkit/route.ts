import {
    CopilotRuntime,
    //GoogleGenerativeAIAdapter,
    copilotRuntimeNextJSAppRouterEndpoint,
    EmptyAdapter,
} from '@copilotkit/runtime';
import { NextRequest } from 'next/server';

import { getEnv } from '@/lib/env.server';

export const POST = async (req: NextRequest) => {
    const env = await getEnv();

    // const serviceAdapter = new GoogleGenerativeAIAdapter({
    //     apiKey: env.AI_API_KEY,
    //     model: 'gemini-2.5-flash',
    // });
    const serviceAdapter = new EmptyAdapter();

    const runtime = new CopilotRuntime({
        remoteEndpoints: [
            { url: env.LANGGRAPH_AGENT_URL },
        ],
    });

    const { handleRequest } = copilotRuntimeNextJSAppRouterEndpoint({
        runtime,
        serviceAdapter,
        endpoint: '/api/copilotkit',
    });

    return handleRequest(req);
}