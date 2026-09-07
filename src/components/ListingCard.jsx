import React from 'react';

export default function ListingCard({ listing }) {
  if (!listing) return null;

  return (
    <div className="group cursor-pointer" data-testid={`listing-card-${listing.id || 'new'}`}>
      <div className="relative aspect-[4/3] overflow-hidden rounded-xl bg-gray-200 dark:bg-gray-800 mb-3">
        <img
          src={listing.imageUrl || 'https://placehold.co/600x400?text=Listing'}
          srcSet={`${listing.imageUrl || 'https://placehold.co/300x200?text=Listing'} 300w, ${listing.imageUrl || 'https://placehold.co/600x400?text=Listing'} 600w, ${listing.imageUrl || 'https://placehold.co/900x600?text=Listing'} 900w`}
          sizes="(max-width: 600px) 100vw, (max-width: 900px) 50vw, 33vw"
          alt={listing.title}
          loading="lazy"
          className="object-cover w-full h-full transition duration-300 group-hover:scale-105"
        />
        <button
          className="absolute top-3 right-3 text-gray-500 hover:text-red-500 hover:scale-110 transition drop-shadow-sm"
          aria-label="Save listing"
        >
          <svg className="w-6 h-6 fill-black/50 stroke-white stroke-2" viewBox="0 0 24 24"><path d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 5.42 4.42 3 7.5 3c1.74 0 3.41.81 4.5 2.09C13.09 3.81 14.76 3 16.5 3 19.58 3 22 5.42 22 8.5c0 3.78-3.4 6.86-8.55 11.54L12 21.35z"></path></svg>
        </button>
      </div>
      <div className="flex justify-between items-start">
        <h3 className="font-semibold text-gray-900 dark:text-white truncate pr-2">{listing.title}</h3>
        <div className="flex items-center gap-1 text-sm text-gray-700 dark:text-gray-300">
          <svg className="w-4 h-4 fill-current" viewBox="0 0 24 24"><path d="M12 17.27L18.18 21l-1.64-7.03L22 9.24l-7.19-.61L12 2 9.19 8.63 2 9.24l5.46 4.73L5.82 21z"></path></svg>
          {listing.rating || '4.9'}
        </div>
      </div>
      <p className="text-gray-500 dark:text-gray-400 text-sm truncate">{listing.category || 'Category'}</p>
      <div className="mt-1 flex items-center gap-1 text-gray-900 dark:text-white">
        <span className="font-semibold">${listing.price}</span>
        <span className="text-sm font-normal">night</span>
      </div>
    </div>
  );
}
