import { CameraFeed } from "@/components/ui/CameraFeed";
import { Button } from "@/components/ui/Button";
import { RiskBadge } from "@/components/ui/Badge";
import { DASH_DETECTIONS, DASH_INCIDENTS } from "@/lib/mock-data";

export default function DashboardPage() {
  return (
    <>
      <div className="page-head">
        <div>
          <h1>Industrial Unit 4 — Assembly Line B</h1>
          <p>Real-time computer vision analysis active. Monitoring for PPE compliance and zone integrity.</p>
        </div>
        <div className="page-head-actions">
          <Button variant="ghost">Camera Feeds</Button>
          <Button>Acknowledge All</Button>
        </div>
      </div>

      <div className="dash-grid">
        <div>
          <div className="panel">
            <CameraFeed camId="CAM-04-MAIN" detections={DASH_DETECTIONS} hazardClasses={[]}>
              <div className="feed-hud"><span className="rec">CAM-04-MAIN</span></div>
              <div className="feed-hud-right">FPS: 29.97 | LATENCY: 42ms</div>
              {DASH_DETECTIONS.map((d) => {
                const [x1, y1, x2, y2] = d.bbox;
                return (
                  <div
                    key={d.id}
                    className={`fbox ${d.safe ? "safe-obj" : "hazard"}`}
                    style={{ left: `${x1}%`, top: `${y1}%`, width: `${x2 - x1}%`, height: `${y2 - y1}%` }}
                  >
                    <div className="fbox-tag">{d.label}</div>
                  </div>
                );
              })}
            </CameraFeed>
          </div>

          <div className="dash-stats">
            <div className="panel">
              <div className="label" style={{ marginBottom: 12 }}>Safety Compliance</div>
              <div style={{ display: "flex", gap: 12 }}>
                <div className="stat-card" style={{ flex: 1 }}>
                  <div className="stat-label">PPE Compliance</div>
                  <div className="stat-value" style={{ color: "var(--safe)" }}>94%</div>
                </div>
                <div className="stat-card" style={{ flex: 1 }}>
                  <div className="stat-label">Zone Violations</div>
                  <div className="stat-value" style={{ color: "var(--high)" }}>
                    1 <span style={{ fontSize: 11, color: "var(--ink-faint)" }}>last hr</span>
                  </div>
                </div>
              </div>
            </div>
            <div className="panel">
              <div className="label" style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
                Active Machinery Alerts <RiskBadge level="Critical" />
              </div>
              <div style={{ fontSize: 12.5, color: "var(--ink-dim)", lineHeight: 1.5 }}>
                <strong style={{ color: "var(--ink)" }}>1</strong> ROBOT_ARM_02 operating near restricted boundary limits.
              </div>
            </div>
          </div>
        </div>

        <div className="alert-card">
          <h3>
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} width={16} height={16}>
              <path d="M12 9v4M12 17h.01M10.3 3.9L1.8 18a2 2 0 001.7 3h17a2 2 0 001.7-3L13.7 3.9a2 2 0 00-3.4 0z" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
            Critical Risk Detected
          </h3>
          <h4>Reasoning</h4>
          <p>Personnel detected within 2 meters of active robotic arm without required Class E hard hat protection. Site Policy IND-202 violation.</p>
          <h4>Potential Outcome</h4>
          <p>High risk of blunt force trauma or collision.</p>
          <h4>Recommended Action</h4>
          <p style={{ marginBottom: 12 }}>Halt robotic arm operation immediately and notify Floor Supervisor.</p>
          <Button variant="danger" style={{ width: "100%" }}>⏻ Halt Machinery</Button>
        </div>
      </div>

      <div className="panel" style={{ marginTop: 18 }}>
        <div className="label" style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
          Recent Manufacturing Incidents
          <a href="#" style={{ color: "var(--accent-2)", fontSize: 12, textTransform: "none", letterSpacing: 0, fontWeight: 600 }}>View All Logs</a>
        </div>
        <div>
          {DASH_INCIDENTS.map(([ts, tag, text]) => (
            <div className="incident-row" key={ts}>
              <span className="ts mono">{ts}</span>
              <strong style={{ width: 110, flexShrink: 0, fontSize: 12 }}>{tag}</strong>
              <span className="txt">{text}</span>
            </div>
          ))}
        </div>
      </div>
    </>
  );
}