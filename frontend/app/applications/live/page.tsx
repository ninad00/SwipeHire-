"use client";

import React, { useEffect, useState, useRef } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import { useAuth } from "@/context/AuthContext";
import { ProtectedRoute } from "@/components/ProtectedRoute";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
    ArrowLeft,
    BrainCircuit,
    Monitor,
    Globe,
    Play,
    CheckCircle,
    XCircle,
    Loader2,
    Activity,
    Terminal,
    ExternalLink,
    ChevronRight,
    XOctagon
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

interface StepData {
    step: number;
    url: string;
    screenshot: string;
    thinking: string;
    next_goal: string;
    action: string;
    message: string;
    timestamp: string;
}

type AppStatus = "connecting" | "running" | "success" | "error";

export default function LiveApplicationPage() {
    return (
        <ProtectedRoute>
            <LiveApplicationContent />
        </ProtectedRoute>
    );
}

function LiveApplicationContent() {
    const searchParams = useSearchParams();
    const router = useRouter();
    const { user, getIdToken } = useAuth();

    const jobId = searchParams.get("jobId");
    const wsUrl = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000";
    const BACKEND_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

    const [status, setStatus] = useState<AppStatus>("connecting");
    const [logs, setLogs] = useState<string[]>([]);
    const [steps, setSteps] = useState<StepData[]>([]);
    const [currentStep, setCurrentStep] = useState<number>(0);
    const [currentScreenshot, setCurrentScreenshot] = useState<string | null>(null);
    const [currentThinking, setCurrentThinking] = useState<string>("");
    const [currentGoal, setCurrentGoal] = useState<string>("");
    const [currentAction, setCurrentAction] = useState<string>("");
    const [activeUrl, setActiveUrl] = useState<string>("");
    const [errorMessage, setErrorMessage] = useState<string>("");
    const [finalResult, setFinalResult] = useState<string>("");

    const logsEndRef = useRef<HTMLDivElement>(null);
    const wsRef = useRef<WebSocket | null>(null);

    // Scroll logs to bottom
    useEffect(() => {
        logsEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [logs, currentThinking]);

    useEffect(() => {
        if (!jobId) {
            setStatus("error");
            setErrorMessage("No Job ID provided. Please select a job from the matches page.");
            return;
        }

        let isMounted = true;
        let ws: WebSocket | null = null;

        async function initWebSocket() {
            try {
                const token = await getIdToken();
                if (!isMounted) return;
                if (!token) {
                    if (isMounted) {
                        setStatus("error");
                        setErrorMessage("Authentication failed. Please log in again.");
                    }
                    return;
                }

                // Connect to FastAPI ws
                const cleanWsUrl = wsUrl.replace(/^http/, "ws");
                const url = `${cleanWsUrl}/ws/apply?token=${encodeURIComponent(token)}&job_id=${encodeURIComponent(jobId!)}`;

                ws = new WebSocket(url);
                wsRef.current = ws;

                ws.onopen = () => {
                    if (isMounted) {
                        setStatus("connecting");
                        setLogs((prev) => [...prev, "Connected to server. Initiating session..."]);
                    }
                };

                ws.onmessage = (event) => {
                    if (!isMounted) return;
                    const data = JSON.parse(event.data);

                    if (data.type === "status") {
                        setLogs((prev) => [...prev, data.message]);
                    }
                    else if (data.type === "error") {
                        setStatus("error");
                        setErrorMessage(data.message);
                        setLogs((prev) => [...prev, `❌ Error: ${data.message}`]);
                    }
                    else if (data.type === "step") {
                        setStatus("running");
                        setCurrentStep(data.step);
                        setActiveUrl(data.url);

                        // Fallback for screenshot if websocket data doesn't have it
                        const screenshotVal = data.screenshot || (user?.uid ? `${BACKEND_URL}/live-screenshot/${user.uid}?t=${Date.now()}` : null);
                        setCurrentScreenshot(screenshotVal);

                        setCurrentThinking(data.thinking);
                        setCurrentGoal(data.next_goal);
                        setCurrentAction(data.action);

                        const newStep: StepData = {
                            step: data.step,
                            url: data.url,
                            screenshot: screenshotVal || "",
                            thinking: data.thinking,
                            next_goal: data.next_goal,
                            action: data.action,
                            message: data.message,
                            timestamp: new Date().toLocaleTimeString()
                        };

                        setSteps((prev) => [...prev, newStep]);
                        setLogs((prev) => [...prev, `Step ${data.step} completed. URL: ${data.url}`]);
                    }
                    else if (data.type === "success") {
                        setStatus("success");
                        setFinalResult(data.result);
                        setLogs((prev) => [...prev, "🎉 Application form automation complete!"]);
                    }
                };

                ws.onclose = (event) => {
                    if (!isMounted) return;
                    if (status !== "success" && status !== "error") {
                        setStatus("error");
                        setErrorMessage("WebSocket connection closed unexpectedly.");
                    }
                };

                ws.onerror = (error) => {
                    if (!isMounted) return;
                    console.error("WS Error:", error);
                    setStatus("error");
                    setErrorMessage("Failed to connect to the auto-apply server.");
                };

            } catch (err: any) {
                if (isMounted) {
                    setStatus("error");
                    setErrorMessage(err.message || "An unexpected error occurred.");
                }
            }
        }

        initWebSocket();

        return () => {
            isMounted = false;
            if (ws) {
                ws.close();
            }
        };
    }, [jobId, getIdToken, wsUrl, user?.uid, BACKEND_URL]);

    // Polling screenshot fallback every 5 seconds when running
    useEffect(() => {
        if (status !== "running" || !user?.uid) return;

        const interval = setInterval(() => {
            const imgUrl = `${BACKEND_URL}/live-screenshot/${user.uid}?t=${Date.now()}`;
            const img = new Image();
            img.onload = () => {
                setCurrentScreenshot(imgUrl);
            };
            img.src = imgUrl;
        }, 5000);

        return () => clearInterval(interval);
    }, [status, user?.uid, BACKEND_URL]);

    const handleBack = () => {
        router.push("/matches");
    };

    const handleAbort = () => {
        if (wsRef.current && (wsRef.current.readyState === WebSocket.OPEN || wsRef.current.readyState === WebSocket.CONNECTING)) {
            try {
                wsRef.current.send(JSON.stringify({ type: "abort" }));
            } catch (e) {
                console.error("Failed to send abort command:", e);
            }
            wsRef.current.close();
        }
        setStatus("error");
        setErrorMessage("Application process aborted by user.");
        setLogs((prev) => [...prev, "🛑 User clicked Abort. Terminating automation..."]);
    };

    return (
        <div className="min-h-[calc(100vh-4rem)] bg-background text-foreground py-6 px-4 md:px-8">
            {/* Top Breadcrumb Header */}
            <div className="max-w-7xl mx-auto flex items-center justify-between mb-8">
                <Button
                    variant="ghost"
                    onClick={handleBack}
                    className="text-muted-foreground hover:text-foreground flex items-center gap-2 hover:bg-muted"
                >
                    <ArrowLeft className="w-4 h-4" />
                    Back to Matches
                </Button>

                {/* Live Pulse Badge & Abort Button */}
                <div className="flex items-center gap-3">
                    {status === "connecting" && (
                        <Badge variant="outline" className="bg-yellow-500/10 text-yellow-500 border-yellow-500/20 px-3 py-1 flex items-center gap-2 animate-pulse">
                            <Loader2 className="w-3 h-3 animate-spin" />
                            Connecting
                        </Badge>
                    )}
                    {status === "running" && (
                        <Badge variant="outline" className="bg-emerald-500/10 text-emerald-400 border-emerald-500/20 px-3 py-1 flex items-center gap-2">
                            <span className="relative flex h-2 w-2">
                                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                                <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
                            </span>
                            Form Filling In Progress
                        </Badge>
                    )}
                    {status === "success" && (
                        <Badge variant="outline" className="bg-green-500/10 text-green-400 border-green-500/20 px-3 py-1 flex items-center gap-2">
                            <CheckCircle className="w-3.5 h-3.5" />
                            Applied Successfully
                        </Badge>
                    )}
                    {status === "error" && (
                        <Badge variant="outline" className="bg-red-500/10 text-red-400 border-red-500/20 px-3 py-1 flex items-center gap-2">
                            <XCircle className="w-3.5 h-3.5" />
                            Failed
                        </Badge>
                    )}

                    {(status === "running" || status === "connecting") && (
                        <Button
                            variant="outline"
                            onClick={handleAbort}
                            className="text-red-400 border-red-500/20 hover:bg-red-500/20 hover:text-red-300 hover:border-red-500/50 transition-all duration-300 flex items-center gap-2 h-9"
                        >
                            <XOctagon className="w-4 h-4" />
                            Abort Run
                        </Button>
                    )}
                </div>
            </div>

            <div className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
                {/* LEFT COLUMN: Controls, Thoughts, Logs */}
                <div className="lg:col-span-5 space-y-6">

                    {/* Agent Brain Card */}
                    <div className="bg-card/40 backdrop-blur-md border border-border rounded-2xl p-6 shadow-xl relative overflow-hidden">
                        <div className="absolute top-0 right-0 p-3 opacity-10">
                            <BrainCircuit className="w-24 h-24 text-primary" />
                        </div>

                        <div className="flex items-center gap-3 mb-4 border-b border-border/50 pb-3">
                            <div className="p-2 bg-primary/10 rounded-lg text-primary">
                                <BrainCircuit className="w-5 h-5" />
                            </div>
                            <div>
                                <h3 className="font-semibold text-lg">Agent Thought Process</h3>
                                <p className="text-xs text-muted-foreground">Real-time reasoning log</p>
                            </div>
                        </div>

                        <div className="space-y-4 min-h-[140px] flex flex-col justify-between">
                            {status === "connecting" && (
                                <div className="flex-1 flex flex-col items-center justify-center py-6 text-center text-muted-foreground">
                                    <Loader2 className="w-8 h-8 animate-spin text-primary mb-2" />
                                    <p className="text-sm">Waiting for agent to initialize...</p>
                                </div>
                            )}

                            {status === "error" && (
                                <div className="flex-1 flex flex-col items-center justify-center py-6 text-center text-red-400">
                                    <XCircle className="w-8 h-8 text-red-500 mb-2" />
                                    <p className="font-medium text-sm">Execution Stopped</p>
                                    <p className="text-xs text-muted-foreground mt-1 max-w-[280px]">{errorMessage}</p>
                                </div>
                            )}

                            {status === "success" && (
                                <div className="flex-1 flex flex-col items-center justify-center py-6 text-center text-green-400">
                                    <CheckCircle className="w-10 h-10 text-green-500 mb-2" />
                                    <p className="font-semibold text-base">Application Submitted!</p>
                                    <p className="text-xs text-muted-foreground mt-1 max-w-[320px]">
                                        The form has been successfully completed and sent.
                                    </p>
                                    {finalResult && (
                                        <div className="mt-4 p-3 bg-green-500/10 border border-green-500/20 rounded-xl text-left text-xs text-green-300 w-full max-h-[120px] overflow-y-auto">
                                            {finalResult}
                                        </div>
                                    )}
                                </div>
                            )}

                            {status === "running" && (
                                <div className="space-y-4">
                                    {/* Current Thought */}
                                    <div>
                                        <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-1.5 flex items-center gap-1.5">
                                            <ChevronRight className="w-3.5 h-3.5 text-primary" />
                                            Active Thought
                                        </h4>
                                        <p className="text-sm font-medium bg-muted/50 border border-border/30 rounded-xl p-3.5 text-foreground leading-relaxed">
                                            {currentThinking || "Deciding next step..."}
                                        </p>
                                    </div>

                                    {/* Next Goal */}
                                    {currentGoal && (
                                        <div>
                                            <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-1.5 flex items-center gap-1.5">
                                                <ChevronRight className="w-3.5 h-3.5 text-indigo-400" />
                                                Next Objective
                                            </h4>
                                            <p className="text-sm bg-indigo-500/5 border border-indigo-500/10 rounded-xl p-3 text-indigo-300">
                                                {currentGoal}
                                            </p>
                                        </div>
                                    )}

                                    {/* Action Taken */}
                                    {currentAction && (
                                        <div>
                                            <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-1 flex items-center gap-1.5">
                                                <ChevronRight className="w-3.5 h-3.5 text-emerald-400" />
                                                Action Executed
                                            </h4>
                                            <div className="text-xs font-mono bg-emerald-500/5 border border-emerald-500/10 rounded-lg p-2.5 text-emerald-400 overflow-x-auto max-w-full">
                                                {currentAction}
                                            </div>
                                        </div>
                                    )}
                                </div>
                            )}

                            {/* Step info row */}
                            {status === "running" && (
                                <div className="flex items-center justify-between text-xs text-muted-foreground border-t border-border/50 pt-4 mt-2">
                                    <div className="flex items-center gap-1">
                                        <Activity className="w-3.5 h-3.5 text-emerald-400 animate-pulse" />
                                        <span>Step {currentStep}</span>
                                    </div>
                                    <span>Real-time stream active</span>
                                </div>
                            )}
                        </div>
                    </div>

                    {/* Console Terminal Logs Card */}
                    <div className="bg-card/40 backdrop-blur-md border border-border rounded-2xl p-5 shadow-xl">
                        <div className="flex items-center gap-2 mb-3 text-muted-foreground text-xs font-semibold uppercase tracking-wider">
                            <Terminal className="w-4 h-4 text-primary" />
                            <span>Console Log Output</span>
                        </div>

                        <div className="bg-black/40 border border-white/5 rounded-xl p-3.5 h-[220px] overflow-y-auto font-mono text-[11px] text-zinc-300 space-y-1.5">
                            {logs.map((log, idx) => (
                                <div key={idx} className="leading-relaxed break-words">
                                    <span className="text-zinc-500 select-none mr-2">[{idx + 1}]</span>
                                    {log.startsWith("❌") ? (
                                        <span className="text-red-400">{log}</span>
                                    ) : log.startsWith("🎉") || log.startsWith("Step") ? (
                                        <span className="text-emerald-400">{log}</span>
                                    ) : (
                                        log
                                    )}
                                </div>
                            ))}
                            <div ref={logsEndRef} />
                        </div>
                    </div>

                </div>

                {/* RIGHT COLUMN: Browser Screen Simulator */}
                <div className="lg:col-span-7">

                    {/* Simulated Browser Frame */}
                    <div className="bg-card/45 backdrop-blur-md border border-border rounded-2xl shadow-2xl overflow-hidden flex flex-col">

                        {/* Header Address Bar Area */}
                        <div className="bg-muted/70 px-4 py-3 border-b border-border/80 flex items-center gap-3">
                            {/* Browser control window dots */}
                            <div className="flex items-center gap-1.5 shrink-0">
                                <span className="w-3 h-3 rounded-full bg-red-500/80"></span>
                                <span className="w-3 h-3 rounded-full bg-yellow-500/80"></span>
                                <span className="w-3 h-3 rounded-full bg-green-500/80"></span>
                            </div>

                            {/* Address bar input */}
                            <div className="flex-1 bg-background/50 border border-border/80 rounded-lg px-3 py-1 flex items-center gap-2 text-xs text-muted-foreground min-w-0">
                                <Globe className="w-3.5 h-3.5 text-primary shrink-0" />
                                <span className="truncate flex-1 select-all">{activeUrl || "https://..."}</span>
                            </div>
                        </div>

                        {/* Simulated Frame Screen Area */}
                        <div className="relative aspect-[16/10] bg-zinc-950 w-full flex items-center justify-center overflow-hidden">
                            <AnimatePresence mode="wait">
                                {currentScreenshot ? (
                                    <motion.img
                                        key={currentScreenshot}
                                        src={currentScreenshot}
                                        alt="Browser live view"
                                        className="w-full h-full object-contain object-top"
                                        initial={{ opacity: 0.3 }}
                                        animate={{ opacity: 1 }}
                                        exit={{ opacity: 0.9 }}
                                        transition={{ duration: 0.2 }}
                                    />
                                ) : (
                                    <div className="text-center text-muted-foreground space-y-3 px-4">
                                        <Monitor className="w-12 h-12 mx-auto text-muted-foreground/30 animate-pulse" />
                                        <div>
                                            <p className="text-sm font-medium">Browser Stream Offline</p>
                                            <p className="text-xs text-muted-foreground/60 max-w-[280px] mx-auto mt-1">
                                                {status === "connecting"
                                                    ? "Starting browser engine and navigating..."
                                                    : status === "error"
                                                        ? "Execution terminated due to an error."
                                                        : "Initializing Playwright browser session..."}
                                            </p>
                                        </div>
                                    </div>
                                )}
                            </AnimatePresence>

                            {/* Status overlay (during execution) */}
                            {status === "running" && (
                                <div className="absolute bottom-4 right-4 bg-black/60 backdrop-blur-md border border-white/10 rounded-lg px-3 py-1.5 text-[10px] text-white flex items-center gap-1.5 font-mono select-none pointer-events-none">
                                    <span className="relative flex h-2 w-2">
                                        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75"></span>
                                        <span className="relative inline-flex rounded-full h-2 w-2 bg-red-500"></span>
                                    </span>
                                    LIVE FEED
                                </div>
                            )}
                        </div>

                        {/* Action Bar / Meta */}
                        <div className="bg-muted/30 px-6 py-4 border-t border-border/80 flex items-center justify-between text-xs text-muted-foreground">
                            <div className="flex items-center gap-2">
                                <Monitor className="w-4 h-4 text-primary" />
                                <span>Simulated Viewport</span>
                            </div>
                            <span className="font-mono text-[10px] bg-muted/80 px-2 py-0.5 rounded border border-border/40 select-none">1280 x 800</span>
                        </div>

                    </div>

                </div>
            </div>
        </div>
    );
}