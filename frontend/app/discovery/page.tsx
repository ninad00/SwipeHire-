"use client";

import { useState, useCallback, useEffect, useRef } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { JobCard } from "@/components/discovery/JobCard";
import { JobDetailsModal } from "@/components/discovery/JobDetailsModal";
import { JobChatbot } from "@/components/discovery/JobChatbot";
import { DatabaseJob } from "@/lib/resumeApi";
import { Briefcase, Heart, X, ChevronUp } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useRouter } from "next/navigation";
import { ProtectedRoute } from "@/components/ProtectedRoute";
import { useAuth } from "@/context/AuthContext";

type WSMessage =
  | { type: "JOB"; job: DatabaseJob }
  | { type: "END" };

const BACKEND_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function DiscoveryPage() {
  const router = useRouter();
  const { getIdToken } = useAuth();

  const [jobs, setJobs] = useState<DatabaseJob[]>([]);
  const [loading, setLoading] = useState(true);
  const socketRef = useRef<WebSocket | null>(null);
  const isFetchingRef = useRef(false);
  const hasLoadedRef = useRef(false);

  const [selectedJob, setSelectedJob] = useState<DatabaseJob | null>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [exitDirection, setExitDirection] = useState<"left" | "right" | null>(null);
  const [pendingModalJob, setPendingModalJob] = useState<DatabaseJob | null>(null);

  // Initial load
  useEffect(() => {
    if (hasLoadedRef.current) return;
    hasLoadedRef.current = true;

    async function loadInitialJobs() {
      const token = await getIdToken();
      if (!token) {
        setLoading(false);
        return;
      }

      try {
        const res = await fetch(`${BACKEND_URL}/save-profile`, {
          headers: { Authorization: `Bearer ${token}` },
        });
        const data = await res.json();
        setJobs(data.ranked_jobs || []);
      } catch (error) {
        console.error("Failed to load initial jobs:", error);
      } finally {
        setLoading(false);
      }
    }

    loadInitialJobs();
  }, [getIdToken]);

  // WebSocket for next jobs
  useEffect(() => {
    if (jobs.length === 0 || loading) return;
    let ws: WebSocket;

    async function connectWS() {
      const token = await getIdToken();
      if (!token) return;

      const wsUrl = BACKEND_URL.replace(/^https?/, (m) => m === "https" ? "wss" : "ws");
      ws = new WebSocket(`${wsUrl}/ws/jobs?token=${token}`);
      socketRef.current = ws;

      ws.onmessage = (event) => {
        const msg: WSMessage = JSON.parse(event.data);
        if (msg.type === "JOB") {
          setJobs((prev) => [...prev, msg.job]);
          isFetchingRef.current = false;
        }
      };
    }

    connectWS();
    return () => ws?.close();
  }, [getIdToken, jobs.length > 0, loading]);

  const requestNextJob = () => {
    if (isFetchingRef.current) return;
    isFetchingRef.current = true;
    socketRef.current?.send(JSON.stringify({ type: "NEXT_JOB" }));
  };

  const currentJob = jobs[0];

  // Handle swipe action
  const handleSwipe = useCallback(
    async (direction: "left" | "right", fromModal = false) => {
      // If called from modal, just close modal and move to next job
      if (fromModal) {
        setIsModalOpen(false);
        setSelectedJob(null);
        setPendingModalJob(null);

        // Move to next job
        setJobs((prev) => prev.slice(1));
        requestNextJob();
        return;
      }

      if (!currentJob) return;

      setExitDirection(direction);

      if (direction === "right") {
        try {
          const token = await getIdToken();

          await fetch(`${BACKEND_URL}/match`, {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              Authorization: `Bearer ${token}`,
            },
            body: JSON.stringify({
              job_id: currentJob.id,
              score: currentJob.score || 0,
            }),
          });

          console.log("Match saved!");
        } catch (err) {
          console.error("Failed to save match:", err);
        }

        setPendingModalJob(currentJob);
      }

      if (direction === "left") {
        try {
          const token = await getIdToken();

          await fetch(`${BACKEND_URL}/not-match`, {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              Authorization: `Bearer ${token}`,
            },
            body: JSON.stringify({
              job_id: currentJob.id,
            }),
          });

          console.log("Skip recorded!");
        } catch (err) {
          console.error("Failed to record skip:", err);
        }
      }

      // Wait for swipe animation to complete, then update
      setTimeout(() => {
        setJobs((prev) => prev.slice(1));
        requestNextJob();
        setExitDirection(null);

        // Show modal AFTER animation completes (only for right swipe)
        if (direction === "right" && pendingModalJob) {
          setSelectedJob(pendingModalJob);
          setIsModalOpen(true);
          setPendingModalJob(null);
        }
      }, 350);
    },
    [currentJob, getIdToken, pendingModalJob]
  );

  // Effect to show modal after animation
  useEffect(() => {
    if (pendingModalJob && !exitDirection) {
      setSelectedJob(pendingModalJob);
      setIsModalOpen(true);
      setPendingModalJob(null);
    }
  }, [pendingModalJob, exitDirection]);

  // Keyboard navigation - only when modal is closed
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (isModalOpen) return;
      if (e.key === "ArrowLeft") handleSwipe("left");
      if (e.key === "ArrowRight") handleSwipe("right");
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [handleSwipe, isModalOpen]);

  const handleViewDetails = useCallback(() => {
    if (!currentJob) return;
    setSelectedJob(currentJob);
    setIsModalOpen(true);
  }, [currentJob]);

  // Modal actions - Save the job and move to next
  const handleModalSave = useCallback(async () => {
    if (!selectedJob) {
      setIsModalOpen(false);
      setSelectedJob(null);
      return;
    }

    // Save match
    try {
      const token = await getIdToken();
      await fetch(`${BACKEND_URL}/match`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          job_id: selectedJob.id,
          score: selectedJob.score || 0
        }),
      });
      console.log("Match saved from modal!");
    } catch (err) {
      console.error("Failed to save match:", err);
    }

    setIsModalOpen(false);
    setSelectedJob(null);
    // Move to next job
    setJobs((prev) => prev.slice(1));
    requestNextJob();
  }, [selectedJob, getIdToken]);

  const handleModalPass = useCallback(() => {
    setIsModalOpen(false);
    setSelectedJob(null);
    // Move to next job (pass)
    setJobs((prev) => prev.slice(1));
    requestNextJob();
  }, []);

  // Loading state
  if (loading) {
    return (
      <div className="h-[calc(100vh-4rem)] bg-background flex items-center justify-center">
        <div className="text-center space-y-4">
          <div className="relative w-20 h-20 mx-auto">
            <div className="absolute inset-0 border-4 border-primary/30 rounded-full animate-ping"></div>
            <div className="absolute inset-0 border-4 border-primary rounded-full border-t-transparent animate-spin"></div>
          </div>
          <p className="text-muted-foreground font-medium animate-pulse">Loading your matches...</p>
        </div>
      </div>
    );
  }

  // No jobs state
  if (jobs.length === 0) {
    return (
      <div className="h-[calc(100vh-4rem)] bg-background flex items-center justify-center">
        <div className="text-center max-w-md px-4">
          <div className="w-24 h-24 mx-auto mb-6 rounded-full bg-primary/20 flex items-center justify-center border border-primary/30">
            <Briefcase className="w-12 h-12 text-primary" />
          </div>
          <h2 className="text-2xl font-bold text-foreground mb-3">No Jobs Yet</h2>
          <p className="text-muted-foreground mb-6">
            Upload your resume to get personalized job recommendations.
          </p>
          <Button onClick={() => router.push("/profile")} className="bg-primary hover:bg-primary/90">
            Upload Resume
          </Button>
        </div>
      </div>
    );
  }

  return (
    <ProtectedRoute>
      <div className="h-[calc(100vh-4rem)] bg-background overflow-hidden">
        <div className="h-full flex gap-2">

          {/* Left - Phone Mockup (30% width, edge-to-edge) */}
          <div className="w-[30%] min-w-[320px] flex items-center justify-center pl-4">
            <div className="relative w-full max-w-[380px]">
              {/* Phone Frame */}
              <div className="relative bg-slate-800 rounded-[2.5rem] p-2.5 shadow-2xl shadow-black/50 border border-slate-700">
                {/* Notch */}
                <div className="absolute top-0 left-1/2 -translate-x-1/2 w-28 h-6 bg-slate-800 rounded-b-xl z-20"></div>

                {/* Screen */}
                <div className="relative bg-gradient-to-b from-slate-900 to-slate-950 rounded-[2rem] overflow-hidden h-[calc(100vh-10rem)] min-h-[500px] max-h-[700px]">
                  {/* Card Container */}
                  <div className="absolute inset-0 pt-6 pb-20 px-3 flex items-center justify-center">
                    <AnimatePresence mode="popLayout">
                      {currentJob && (
                        <JobCard
                          key={currentJob.id}
                          job={currentJob}
                          exitDirection={exitDirection}
                          onSwipe={handleSwipe}
                          onViewDetails={handleViewDetails}
                        />
                      )}
                    </AnimatePresence>
                  </div>

                  {/* Bottom Actions */}
                  <div className="absolute bottom-0 left-0 right-0 h-16 bg-gradient-to-t from-black/80 to-transparent flex items-center justify-center gap-4">
                    <motion.button
                      whileHover={{ scale: 1.1 }}
                      whileTap={{ scale: 0.9 }}
                      onClick={() => handleSwipe("left")}
                      className="w-12 h-12 rounded-full bg-red-500/20 border-2 border-red-400 flex items-center justify-center text-red-400 hover:bg-red-500/30 transition-colors"
                    >
                      <X className="w-6 h-6" />
                    </motion.button>

                    <motion.button
                      whileHover={{ scale: 1.1 }}
                      whileTap={{ scale: 0.9 }}
                      onClick={handleViewDetails}
                      className="w-9 h-9 rounded-full bg-primary/20 border-2 border-primary flex items-center justify-center text-primary hover:bg-primary/30 transition-colors"
                    >
                      <ChevronUp className="w-4 h-4" />
                    </motion.button>

                    <motion.button
                      whileHover={{ scale: 1.1 }}
                      whileTap={{ scale: 0.9 }}
                      onClick={() => handleSwipe("right")}
                      className="w-12 h-12 rounded-full bg-emerald-500/20 border-2 border-emerald-400 flex items-center justify-center text-emerald-400 hover:bg-emerald-500/30 transition-colors"
                    >
                      <Heart className="w-6 h-6" />
                    </motion.button>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Right - AI Chatbot (70% width, edge-to-edge) */}
          <div className="flex-1 pr-4">
            <JobChatbot currentJob={currentJob} />
          </div>

        </div>

        <JobDetailsModal
          job={selectedJob}
          isOpen={isModalOpen}
          onClose={handleModalPass}
          onApply={handleModalSave}
          onPass={handleModalPass}
        />
      </div>
    </ProtectedRoute>
  );
}