"use client";

import { CopilotKit } from "@copilotkit/react-core";
import { CopilotChat } from "@copilotkit/react-ui";
import "@copilotkit/react-ui/styles.css";

export default function Home() {
    return (
      <CopilotKit runtimeUrl="/api/copilotkit" agent="offers_agent">
        <CopilotChat />
      </CopilotKit>
    );
}