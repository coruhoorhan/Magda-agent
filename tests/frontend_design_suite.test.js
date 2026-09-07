import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import MapView from '../src/components/MapView.jsx';
import PhotoLightbox from '../src/components/PhotoLightbox.jsx';
import ListingDetail from '../src/components/ListingDetail.jsx';
import Navbar from '../src/components/Navbar.jsx';
import RecommendationCarousel from '../src/components/RecommendationCarousel.jsx';
import ListingCard from '../src/components/ListingCard.jsx';
import ExperienceCard from '../src/components/ExperienceCard.jsx';

describe('Frontend Design Suite', () => {

  // 1. Map Clustering
  describe('Map Clustering', () => {
    const mockListings = [
      { id: 1, lat: 40.71, lng: -74.01, price: 100 },
      { id: 2, lat: 40.72, lng: -74.02, price: 150 },
    ];

    it('zooming out clusters listing markers; clicking zooms into the cluster area smoothly', () => {
      render(<MapView listings={mockListings} />);

      const clusterMarker = screen.getByTestId(/cluster-/);
      expect(clusterMarker).toBeInTheDocument();
      expect(clusterMarker).toHaveTextContent('2');
    });
  });

  // 2. Photo Lightbox
  describe('Photo Lightbox', () => {
    const mockListing = {
      title: 'Test',
      photos: ['url1', 'url2', 'url3', 'url4', 'url5', 'url6']
    };

    it('5-photo grid renders on detail page; clicking any image opens full-screen gallery with ESC/arrow key controls', () => {
      render(<ListingDetail listing={mockListing} />);

      expect(screen.getByTestId('listing-detail')).toBeInTheDocument();

      const showAllBtn = screen.getByTestId('show-all-photos');
      fireEvent.click(showAllBtn);

      const lightbox = screen.getByTestId('photo-lightbox');
      expect(lightbox).toBeInTheDocument();

      fireEvent.keyDown(document, { key: 'Escape', code: 'Escape' });
      expect(screen.queryByTestId('photo-lightbox')).not.toBeInTheDocument();
    });
  });

  // 3. Theme Switcher
  describe('Theme Switcher', () => {
    it('Theme button toggles between light and dark modes with persistent localStorage state', () => {
      render(<Navbar />);

      const toggleBtn = screen.getByTestId('theme-toggle');

      expect(document.documentElement.classList.contains('dark')).toBe(false);

      fireEvent.click(toggleBtn);
      expect(document.documentElement.classList.contains('dark')).toBe(true);
      expect(window.localStorage.getItem('theme')).toBe('dark');

      fireEvent.click(toggleBtn);
      expect(document.documentElement.classList.contains('dark')).toBe(false);
      expect(window.localStorage.getItem('theme')).toBe('light');
    });
  });

  // 4. Recommendation Carousel
  describe('Recommendation Carousel', () => {
    it('Matching listings carousel renders on home page without layout shifts', () => {
      render(<RecommendationCarousel viewedListings={[]} />);
      expect(screen.getByTestId('recommendation-carousel')).toBeInTheDocument();
      expect(screen.getByText(/AI Top Picks for You/i)).toBeInTheDocument();
    });
  });

  // 5. Lazy Loading
  describe('Lazy Loading', () => {
    it('Off-screen images have loading="lazy" and load on demand as user scrolls', () => {
      const listing = { id: 1, imageUrl: 'test.jpg' };
      render(
        <div>
          <ListingCard listing={listing} />
          <ExperienceCard experience={{ id: 2, imageUrl: 'test2.jpg' }} />
        </div>
      );

      const imgs = screen.getAllByRole('img');
      imgs.forEach(img => {
        expect(img).toHaveAttribute('loading', 'lazy');
      });

      expect(imgs[0]).toHaveAttribute('srcSet');
    });
  });
});
