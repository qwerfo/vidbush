import React from "react";
import { useRoute } from "./hooks.js";
import Projects from "./pages/Projects.jsx";
import Project from "./pages/Project.jsx";
import Brands from "./pages/Brands.jsx";
import SettingsPage from "./pages/Settings.jsx";
import Queue from "./pages/Queue.jsx";
import Review from "./pages/Review.jsx";

const NAV = [
  ["projects", "Projects"],
  ["queue", "Queue"],
  ["brands", "Brands"],
  ["settings", "Settings"],
];

export default function App() {
  const [page, arg, sub] = useRoute();
  const active = page === "p" || page === "review" ? "projects" : page;
  let body;
  if (page === "p" && arg) body = <Project id={arg} key={arg} initialShot={sub} />;
  else if (page === "review" && arg) body = <Review id={arg} key={arg} />;
  else if (page === "brands") body = <Brands initial={arg} />;
  else if (page === "settings") body = <SettingsPage />;
  else if (page === "queue") body = <Queue initial={arg} key={arg || "queue"} />;
  else body = <Projects />;
  return (
    <div className="app">
      <header className="topbar">
        <a className="logo" href="#/projects">
          <span className="logo-mark" aria-hidden="true">&#9654;</span> Vidbush
        </a>
        <nav>
          {NAV.map(([k, label]) => (
            <a key={k} href={`#/${k}`} className={active === k ? "active" : ""} aria-current={active === k ? "page" : undefined}>
              {label}
            </a>
          ))}
        </nav>
      </header>
      <main>{body}</main>
    </div>
  );
}
