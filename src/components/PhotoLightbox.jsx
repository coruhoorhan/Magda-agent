import React, { useEffect, useCallback } from 'react';

export default function PhotoLightbox({ photos, initialIndex = 0, onClose }) {
  const [currentIndex, setCurrentIndex] = React.useState(initialIndex);

  const goToNext = useCallback(() => {
    setCurrentIndex((prev) => (prev === photos.length - 1 ? 0 : prev + 1));
  }, [photos.length]);

  const goToPrev = useCallback(() => {
    setCurrentIndex((prev) => (prev === 0 ? photos.length - 1 : prev - 1));
  }, [photos.length]);

  const handleKeyDown = useCallback((e) => {
    if (e.key === 'Escape') onClose();
    if (e.key === 'ArrowRight') goToNext();
    if (e.key === 'ArrowLeft') goToPrev();
  }, [onClose, goToNext, goToPrev]);

  useEffect(() => {
    document.addEventListener('keydown', handleKeyDown);
    document.body.style.overflow = 'hidden';

    return () => {
      document.removeEventListener('keydown', handleKeyDown);
      document.body.style.overflow = '';
    };
  }, [handleKeyDown]);

  if (!photos || photos.length === 0) return null;

  return (
    <div
      className="fixed inset-0 z-50 bg-black/95 flex items-center justify-center p-4"
      data-testid="photo-lightbox"
      aria-modal="true"
      role="dialog"
    >
      <button
        className="absolute top-4 right-4 text-white hover:bg-white/20 p-2 rounded-full transition"
        onClick={onClose}
        aria-label="Close lightbox"
      >
        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12"></path></svg>
      </button>

      <button
        className="absolute left-4 top-1/2 -translate-y-1/2 text-white hover:bg-white/20 p-2 rounded-full transition"
        onClick={goToPrev}
        aria-label="Previous photo"
      >
        <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 19l-7-7 7-7"></path></svg>
      </button>

      <img
        src={photos[currentIndex]}
        alt={`Photo ${currentIndex + 1} of ${photos.length}`}
        className="max-h-full max-w-full object-contain"
      />

      <button
        className="absolute right-4 top-1/2 -translate-y-1/2 text-white hover:bg-white/20 p-2 rounded-full transition"
        onClick={goToNext}
        aria-label="Next photo"
      >
        <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5l7 7-7 7"></path></svg>
      </button>

      <div className="absolute bottom-4 left-1/2 -translate-x-1/2 text-white/80 font-medium tracking-widest text-sm">
        {currentIndex + 1} / {photos.length}
      </div>
    </div>
  );
}
