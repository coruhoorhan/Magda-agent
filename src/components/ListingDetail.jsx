import React, { useState } from 'react';
import PhotoLightbox from './PhotoLightbox.jsx';

export default function ListingDetail({ listing }) {
  const [lightboxOpen, setLightboxOpen] = useState(false);
  const [initialPhotoIndex, setInitialPhotoIndex] = useState(0);

  if (!listing) return null;

  const photos = listing.photos || [];
  const displayPhotos = photos.slice(0, 5);

  const handlePhotoClick = (index) => {
    setInitialPhotoIndex(index);
    setLightboxOpen(true);
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8" data-testid="listing-detail">
      <h1 className="text-3xl font-bold mb-4 dark:text-white">{listing.title}</h1>

      {displayPhotos.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-2 rounded-xl overflow-hidden h-[60vh] relative mb-8">
          <div
            className="md:col-span-2 h-full cursor-pointer relative group"
            onClick={() => handlePhotoClick(0)}
          >
            <img
              src={displayPhotos[0]}
              alt="Main listing photo"
              className="w-full h-full object-cover transition duration-300 group-hover:brightness-90"
              loading="lazy"
            />
          </div>

          <div className="hidden md:grid md:col-span-2 grid-cols-2 grid-rows-2 gap-2 h-full">
            {displayPhotos.slice(1, 5).map((photo, idx) => (
              <div
                key={idx}
                className="w-full h-full cursor-pointer relative group"
                onClick={() => handlePhotoClick(idx + 1)}
              >
                <img
                  src={photo}
                  alt={`Listing photo ${idx + 2}`}
                  className="w-full h-full object-cover transition duration-300 group-hover:brightness-90"
                  loading="lazy"
                />
              </div>
            ))}
          </div>

          <button
            className="absolute bottom-4 right-4 bg-white/90 px-4 py-2 rounded-lg border shadow-sm font-semibold hover:bg-gray-50 flex items-center gap-2 text-sm z-10 dark:text-gray-900"
            onClick={() => handlePhotoClick(0)}
            data-testid="show-all-photos"
          >
            Show all photos
          </button>
        </div>
      )}

      {lightboxOpen && (
        <PhotoLightbox
          photos={photos}
          initialIndex={initialPhotoIndex}
          onClose={() => setLightboxOpen(false)}
        />
      )}

      <div className="mt-8 prose dark:prose-invert">
        <h2 className="dark:text-white">About this space</h2>
        <p className="dark:text-gray-300">{listing.description}</p>
      </div>
    </div>
  );
}
