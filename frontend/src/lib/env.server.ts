"use server";

import { z } from "zod";
import fs from "node:fs";

const envSchema = z.object({
    LANGGRAPH_AGENT_URL: z.string().url(),
    AI_API_KEY_FILE: z.string(),
});

export async function getEnv() {
    const parsed = envSchema.safeParse({
        LANGGRAPH_AGENT_URL: process.env.LANGGRAPH_AGENT_URL,
        AI_API_KEY_FILE: process.env.AI_API_KEY_FILE,
    });

    if (!parsed.success) {
        console.error(
            "Invalid or missing environment variables:",
            parsed.error.flatten().fieldErrors
        );
        throw new Error("Invalid or missing environment variables");
    }

    let aiAPIKey: string;
    try {
        aiAPIKey = fs.readFileSync(parsed.data.AI_API_KEY_FILE, "utf-8").trim();
    } catch (error) {
        console.error("Failed to read AI API key from secret file:", error);
        throw new Error("Failed to read AI API key from secret file");
    }

    return {
        LANGGRAPH_AGENT_URL: parsed.data.LANGGRAPH_AGENT_URL,
        AI_API_KEY: aiAPIKey,
    };
}