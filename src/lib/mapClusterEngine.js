export function getClusters(listings, bounds, zoomLevel) {
  if (zoomLevel >= 13) {
    return listings.map(l => ({ ...l, isCluster: false }));
  }
  const clusters = {};
  const gridSize = 0.5;
  listings.forEach(listing => {
    if (
      bounds &&
      (listing.lat < bounds.minLat ||
        listing.lat > bounds.maxLat ||
        listing.lng < bounds.minLng ||
        listing.lng > bounds.maxLng)
    ) {
      return;
    }
    const gridX = Math.floor(listing.lng / gridSize);
    const gridY = Math.floor(listing.lat / gridSize);
    const clusterId = `${gridX}-${gridY}`;
    if (!clusters[clusterId]) {
      clusters[clusterId] = {
        isCluster: true,
        id: clusterId,
        count: 0,
        latSum: 0,
        lngSum: 0,
        minLat: listing.lat,
        maxLat: listing.lat,
        minLng: listing.lng,
        maxLng: listing.lng,
        listings: [],
      };
    }
    clusters[clusterId].count += 1;
    clusters[clusterId].latSum += listing.lat;
    clusters[clusterId].lngSum += listing.lng;
    clusters[clusterId].minLat = Math.min(clusters[clusterId].minLat, listing.lat);
    clusters[clusterId].maxLat = Math.max(clusters[clusterId].maxLat, listing.lat);
    clusters[clusterId].minLng = Math.min(clusters[clusterId].minLng, listing.lng);
    clusters[clusterId].maxLng = Math.max(clusters[clusterId].maxLng, listing.lng);
    clusters[clusterId].listings.push(listing);
  });
  return Object.values(clusters).map(c => {
    if (c.count === 1) {
      return { ...c.listings[0], isCluster: false };
    }
    return {
      ...c,
      lat: c.latSum / c.count,
      lng: c.lngSum / c.count,
    };
  });
}
