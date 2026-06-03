"use client";

import React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Compass, Heart, User, LogOut, Briefcase } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useAuth } from "@/context/AuthContext";

const Navigation: React.FC = () => {
  const pathname = usePathname();
  const { signOut } = useAuth(); // add this

  // Don't show navigation on login page or landing page
  if (pathname === "/" || pathname === "/login") {
    return null;
  }

  const isActive = (path: string) => pathname === path;

  const handleSignOut = async () => {
    await signOut();
  };

  return (
    <header className="sticky top-0 z-50 bg-card/80 backdrop-blur-md border-b border-border">
      <div className="max-w-7xl mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <Link href="/discovery" className="flex items-center gap-2">
            <Briefcase className="w-8 h-8 text-primary" strokeWidth={1.5} />
            <span className="text-xl font-bold text-foreground">
              SwipeHire
            </span>
          </Link>

          {/* Nav links */}
          <nav className="flex items-center gap-1">
            <Link
              href="/discovery"
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${isActive("/discovery")
                ? "bg-primary/10 text-primary"
                : "text-muted-foreground hover:text-foreground hover:bg-muted"
                }`}
            >
              <Compass className="w-4 h-4" />
              <span className="hidden sm:inline">Discover</span>
            </Link>

            <Link
              href="/matches"
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${isActive("/matches")
                ? "bg-primary/10 text-primary"
                : "text-muted-foreground hover:text-foreground hover:bg-muted"
                }`}
            >
              <Heart className="w-4 h-4" />
              <span className="hidden sm:inline">Matches</span>
            </Link>

            <Link
              href="/applications"
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${isActive("/applications")
                ? "bg-primary/10 text-primary"
                : "text-muted-foreground hover:text-foreground hover:bg-muted"
                }`}
            >
              <Briefcase className="w-4 h-4" />
              <span className="hidden sm:inline">Applications</span>
            </Link>

            <Link
              href="/profile"
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${isActive("/profile")
                ? "bg-primary/10 text-primary"
                : "text-muted-foreground hover:text-foreground hover:bg-muted"
                }`}
            >
              <User className="w-4 h-4" />
              <span className="hidden sm:inline">Profile</span>
            </Link>

            <Button
              variant="ghost"
              size="icon"
              onClick={handleSignOut}
              className="ml-2 text-muted-foreground hover:text-destructive"
            >
              <LogOut className="w-4 h-4" />
            </Button>
          </nav>
        </div>
      </div>
    </header>
  );
};

export default Navigation;