export function getRecommendations(viewedListings = []) {
  const allListings = [
    { id: 101, title: 'Mountain View Cabin', price: 120, category: 'Cabin', imageUrl: 'https://placehold.co/400x300?text=Cabin' },
    { id: 102, title: 'Beachfront Villa', price: 350, category: 'Villa', imageUrl: 'https://placehold.co/400x300?text=Villa' },
    { id: 103, title: 'Urban Loft', price: 90, category: 'Apartment', imageUrl: 'https://placehold.co/400x300?text=Loft' },
    { id: 104, title: 'Cozy Cottage', price: 85, category: 'Cabin', imageUrl: 'https://placehold.co/400x300?text=Cottage' },
    { id: 105, title: 'Luxury Penthouse', price: 500, category: 'Apartment', imageUrl: 'https://placehold.co/400x300?text=Penthouse' }
  ];
  if (!viewedListings || viewedListings.length === 0) {
    return allListings.slice(0, 3);
  }
  const preferredCategories = new Set(viewedListings.map(l => l.category));
  const recommendations = allListings.filter(l => !viewedListings.find(v => v.id === l.id));
  recommendations.sort((a, b) => {
    const aPref = preferredCategories.has(a.category) ? 1 : 0;
    const bPref = preferredCategories.has(b.category) ? 1 : 0;
    return bPref - aPref;
  });
  return recommendations.slice(0, 4);
}
