import React from 'react';
import Navbar from './components/Navbar.jsx';
import MapView from './components/MapView.jsx';
import RecommendationCarousel from './components/RecommendationCarousel.jsx';
import ListingDetail from './components/ListingDetail.jsx';

const mockListing = {
  id: 1,
  title: 'Spectacular Glass House in the Forest',
  description: 'Immerse yourself in nature in this stunning architectural masterpiece. Floor-to-ceiling windows provide 360-degree views of the surrounding old-growth forest. Features include a hot tub, chef\'s kitchen, and a hanging fireplace.',
  price: 450,
  photos: [
    'https://placehold.co/1200x800?text=Photo+1',
    'https://placehold.co/800x600?text=Photo+2',
    'https://placehold.co/800x600?text=Photo+3',
    'https://placehold.co/800x600?text=Photo+4',
    'https://placehold.co/800x600?text=Photo+5',
    'https://placehold.co/800x600?text=Photo+6',
  ]
};

const mapListings = [
  { id: 101, lat: 40.7128, lng: -74.0060, price: 150, category: 'Apartment' },
  { id: 102, lat: 40.7138, lng: -74.0050, price: 200, category: 'Loft' },
  { id: 103, lat: 40.7148, lng: -74.0070, price: 90, category: 'Room' },
  { id: 104, lat: 34.0522, lng: -118.2437, price: 350, category: 'House' }, // Far away
  { id: 105, lat: 40.7300, lng: -73.9900, price: 400, category: 'Apartment' }
];

const mockViewed = [
  { id: 101, category: 'Apartment' }
];

export default function App() {
  return (
    <div className="min-h-screen bg-white dark:bg-gray-900 transition-colors">
      <Navbar />

      <main>
        {/* Listing Detail section with Photo Grid / Lightbox */}
        <section className="border-b dark:border-gray-800 pb-8">
          <ListingDetail listing={mockListing} />
        </section>

        {/* AI Recommendations */}
        <section className="border-b dark:border-gray-800 py-8">
          <RecommendationCarousel viewedListings={mockViewed} />
        </section>

        {/* Map Clustering Demo */}
        <section className="py-8 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <h2 className="text-2xl font-semibold mb-4 dark:text-white">Explore the Map</h2>
          <MapView listings={mapListings} />
        </section>
      </main>
    </div>
  );
}
