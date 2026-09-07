import React, { useState, useEffect } from 'react';
import { getRecommendations } from '../lib/recommendationEngine.js';
import ListingCard from './ListingCard.jsx';

export default function RecommendationCarousel({ viewedListings }) {
  const [recommendations, setRecommendations] = useState([]);

  useEffect(() => {
    const recs = getRecommendations(viewedListings);
    setRecommendations(recs);
  }, [viewedListings]);

  if (!recommendations || recommendations.length === 0) return null;

  return (
    <div className="my-8 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto" data-testid="recommendation-carousel">
      <h2 className="text-2xl font-semibold mb-4 dark:text-white flex items-center gap-2">
        <svg className="w-5 h-5 text-purple-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path></svg>
        AI Top Picks for You
      </h2>
      <div className="flex overflow-x-auto gap-4 pb-4 snap-x no-scrollbar">
        {recommendations.map(listing => (
          <div key={listing.id} className="min-w-[280px] sm:min-w-[320px] snap-start shrink-0">
            <ListingCard listing={listing} />
          </div>
        ))}
      </div>
    </div>
  );
}
