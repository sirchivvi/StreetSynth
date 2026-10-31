import React, { useState, useEffect } from 'react';
import { PageId, BackendStatus } from './types';
import { Navbar } from './components/Navbar';
import { Footer } from './components/Footer';
import { Home } from './pages/Home';
import { Demo } from './pages/Demo';
import { Metrics } from './pages/Metrics';
import { Methodology } from './pages/Methodology';
import { About } from './pages/About';

export const App: React.FC = () => {
  // Sync page state with window.location.hash for deep-linking
  const getInitialPage = (): PageId => {
    const hash = window.location.hash.replace('#', '');
    if (['home', 'demo', 'metrics', 'methodology', 'about'].includes(hash)) {
      return hash as PageId;
    }
    return 'home';
  };

  const [currentPage, setCurrentPage] = useState<PageId>(getInitialPage());
  const [backendStatus, setBackendStatus] = useState<BackendStatus>({
    online: false,
    latency_ms: 0
  });

  // Check Backend Connectivity on mount
  useEffect(() => {
    const checkBackend = async () => {
      const startTime = performance.now();
      const apiBase = (import.meta.env.VITE_API_BASE_URL as string) || '';
      try {
        const res = await fetch(`${apiBase}/api/status`, {
          signal: AbortSignal.timeout(2500)
        });
        if (res.ok) {
          const data = await res.json();
          const latency = Math.round(performance.now() - startTime);
          setBackendStatus({
            online: true,
            device: data.device,
            cuda_available: data.cuda_available,
            checkpoint_available: data.checkpoint_available,
            pipeline_loaded: data.pipeline_loaded,
            models: data.models,
            latency_ms: latency
          });
        } else {
          setBackendStatus({ online: false });
        }
      } catch {
        // Backend offline or running purely as static showcase
        setBackendStatus({ online: false });
      }
    };

    checkBackend();
  }, []);

  // Update hash when page changes
  const handleNavigate = (page: PageId) => {
    setCurrentPage(page);
    window.location.hash = page;
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  // Listen to hash changes (browser back/forward buttons)
  useEffect(() => {
    const onHashChange = () => {
      const hash = window.location.hash.replace('#', '');
      if (['home', 'demo', 'metrics', 'methodology', 'about'].includes(hash)) {
        setCurrentPage(hash as PageId);
      }
    };
    window.addEventListener('hashchange', onHashChange);
    return () => window.removeEventListener('hashchange', onHashChange);
  }, []);

  return (
    <div className="min-h-screen flex flex-col bg-[#101412] text-[#F3F5F4]">
      {/* Top Navigation */}
      <Navbar
        currentPage={currentPage}
        onNavigate={handleNavigate}
        backendStatus={backendStatus}
      />

      {/* Main Content Area */}
      <main className="flex-1">
        {currentPage === 'home' && <Home onNavigate={handleNavigate} />}
        {currentPage === 'demo' && <Demo backendStatus={backendStatus} />}
        {currentPage === 'metrics' && <Metrics />}
        {currentPage === 'methodology' && <Methodology />}
        {currentPage === 'about' && <About />}
      </main>

      {/* Academic Footer */}
      <Footer onNavigate={handleNavigate} />
    </div>
  );
};

export default App;
