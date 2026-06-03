"use client";

import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { MapPin, Building2, Briefcase, X, ExternalLink, Trash2 } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import Link from "next/link";
import { ProtectedRoute } from "@/components/ProtectedRoute";
import { useAuth } from "@/context/AuthContext";
import { JobDetailsModal } from "@/components/discovery/JobDetailsModal";
import { DatabaseJob } from "@/lib/resumeApi";

interface MatchedJob {
  id: string;
  title: string;
  company_name: string;
  location: string;
  description: string;
  extensions?: string[];
  job_highlights?: string[];
  apply_options?: any;
  matched_at?: any;
  score?: number;
}

const BACKEND_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function MatchesPage() {
  const { getIdToken } = useAuth();
  const [matches, setMatches] = useState<MatchedJob[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedJob, setSelectedJob] = useState<DatabaseJob | null>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);

  // Fetch matches from Firebase via backend
  useEffect(() => {
    async function fetchMatches() {
      try {
        const token = await getIdToken();
        if (!token) {
          setLoading(false);
          return;
        }

        const res = await fetch(`${BACKEND_URL}/matches`, {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        });

        if (res.ok) {
          const data = await res.json();
          setMatches(data.matches || []);
        }
      } catch (error) {
        console.error("Failed to fetch matches:", error);
      } finally {
        setLoading(false);
      }
    }

    fetchMatches();
  }, [getIdToken]);

  const handleRemoveMatch = async (jobId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setMatches((prev) => prev.filter((job) => job.id !== jobId));

    try {
      const token = await getIdToken();
      await fetch(`${BACKEND_URL}/match/${jobId}`, {
        method: "DELETE",
        headers: { Authorization: `Bearer ${token}` },
      });
    } catch (error) {
      console.error("Failed to delete match:", error);
    }
  };

  const handleClearAll = async () => {
    setMatches([]);

    try {
      const token = await getIdToken();
      await fetch(`${BACKEND_URL}/matches`, {
        method: "DELETE",
        headers: { Authorization: `Bearer ${token}` },
      });
    } catch (error) {
      console.error("Failed to clear all matches:", error);
    }
  };

  const handleJobClick = (job: MatchedJob) => {
    setSelectedJob(job as DatabaseJob);
    setIsModalOpen(true);
  };

  const handleModalClose = () => {
    setIsModalOpen(false);
    setSelectedJob(null);
  };

  if (loading) {
    return (
      <ProtectedRoute>
        <div className="min-h-screen bg-background flex items-center justify-center">
          <div className="text-center space-y-4">
            <div className="relative w-16 h-16 mx-auto">
              <div className="absolute inset-0 border-4 border-primary rounded-full border-t-transparent animate-spin"></div>
            </div>
            <p className="text-muted-foreground font-medium">Loading your matches...</p>
          </div>
        </div>
      </ProtectedRoute>
    );
  }

  return (
    <ProtectedRoute>
      <div className="min-h-screen ">
        <div className="max-w-6xl mx-auto px-4 py-8">
          {/* Header */}
          <div className="flex items-center justify-between mb-8">
            <div>
              <h1 className="text-3xl font-bold text-foreground">Your Matches</h1>
              <p className="text-muted-foreground mt-1">
                Jobs you're interested in applying to
              </p>
            </div>
            {matches.length > 0 && (
              <Button
                variant="outline"
                onClick={handleClearAll}
                className="text-red-400 border-red-400/50 hover:bg-red-500/20 hover:text-red-300"
              >
                <Trash2 className="w-4 h-4 mr-2" />
                Clear All
              </Button>
            )}
          </div>

          {matches.length > 0 ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              <AnimatePresence mode="popLayout">
                {matches.map((job, index) => (
                  <motion.div
                    key={job.id}
                    layout
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.9 }}
                    transition={{ delay: index * 0.05 }}
                    onClick={() => handleJobClick(job)}
                    className="bg-card backdrop-blur-sm rounded-2xl p-6 border border-border hover:border-primary/50 transition-colors group cursor-pointer"
                  >
                    {/* Header */}
                    <div className="flex items-start gap-3 mb-4">
                      <div className="w-12 h-12 rounded-xl bg-primary/20 flex items-center justify-center shrink-0">
                        <Building2 className="w-6 h-6 text-primary" />
                      </div>
                      <div className="flex-1 min-w-0">
                        <h3 className="font-semibold text-foreground truncate">{job.title}</h3>
                        <p className="text-sm text-muted-foreground">{job.company_name}</p>
                      </div>
                      <Button
                        variant="ghost"
                        size="icon"
                        className="opacity-0 group-hover:opacity-100 transition-opacity text-muted-foreground hover:text-red-400 hover:bg-red-500/20"
                        onClick={(e) => handleRemoveMatch(job.id, e)}
                      >
                        <X className="w-4 h-4" />
                      </Button>
                    </div>

                    {/* Tags */}
                    <div className="flex flex-wrap gap-1.5 mb-4">
                      {(job.extensions || job.job_highlights || []).slice(0, 3).map((tag, i) => (
                        <Badge key={i} variant="outline" className="bg-primary/10 text-primary border-primary/30 text-xs">
                          {tag}
                        </Badge>
                      ))}
                    </div>

                    {/* Location */}
                    <div className="flex items-center gap-2 text-sm text-muted-foreground mb-4">
                      <MapPin className="w-3.5 h-3.5" />
                      <span>{job.location || "Location not specified"}</span>
                    </div>

                    {/* Description Preview */}
                    <p className="text-sm text-muted-foreground/70 line-clamp-2 mb-4">
                      {(job.description || "").replace(/<[^>]*>/g, '').substring(0, 100)}...
                    </p>

                    {/* Action Buttons */}
                    <div className="flex gap-2 mt-auto">
                      <Button
                        variant="outline"
                        className="flex-1 border-primary/30 text-primary hover:bg-primary/10"
                        onClick={(e) => {
                          e.stopPropagation();
                          handleJobClick(job);
                        }}
                      >
                        <ExternalLink className="w-3.5 h-3.5 mr-1.5" />
                        Details
                      </Button>
                      <Button
                        asChild
                        className="flex-1 bg-gradient-to-r from-violet-600 to-indigo-600 hover:from-violet-700 hover:to-indigo-700 text-white border-0 shadow-md transition-all duration-300 hover:scale-[1.02]"
                        onClick={(e) => e.stopPropagation()}
                      >
                        <Link href={`/applications/live?jobId=${job.id}`}>
                          <Briefcase className="w-3.5 h-3.5 mr-1.5" />
                          Auto Apply
                        </Link>
                      </Button>
                    </div>
                  </motion.div>
                ))}
              </AnimatePresence>
            </div>
          ) : (
            <div className="text-center py-16">
              <div className="w-24 h-24 mx-auto mb-6 rounded-full bg-card flex items-center justify-center border border-border">
                <Briefcase className="w-12 h-12 text-muted-foreground" />
              </div>
              <h2 className="text-2xl font-semibold text-foreground mb-2">No matches yet</h2>
              <p className="text-muted-foreground mb-6">
                Start swiping right on jobs you're interested in!
              </p>
              <Button asChild className="bg-primary hover:bg-primary/90">
                <Link href="/discovery">Discover Jobs</Link>
              </Button>
            </div>
          )}
        </div>

        {/* Job Details Modal */}
        <JobDetailsModal
          job={selectedJob}
          isOpen={isModalOpen}
          onClose={handleModalClose}
          onApply={handleModalClose}
          onPass={handleModalClose}
        />
      </div>
    </ProtectedRoute>
  );
}