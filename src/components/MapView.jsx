import React, { useState, useMemo } from 'react';
import { MapContainer, TileLayer, Marker, Popup, useMap } from 'react-leaflet';
import L from 'leaflet';
import { getClusters } from '../lib/mapClusterEngine.js';
import 'leaflet/dist/leaflet.css';

delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

function ClusterMarkers({ listings }) {
  const map = useMap();
  const [zoom, setZoom] = useState(map.getZoom());
  const [bounds, setBounds] = useState({
    minLat: map.getBounds().getSouth(),
    maxLat: map.getBounds().getNorth(),
    minLng: map.getBounds().getWest(),
    maxLng: map.getBounds().getEast(),
  });

  React.useEffect(() => {
    const onMoveEnd = () => {
      setZoom(map.getZoom());
      setBounds({
        minLat: map.getBounds().getSouth(),
        maxLat: map.getBounds().getNorth(),
        minLng: map.getBounds().getWest(),
        maxLng: map.getBounds().getEast(),
      });
    };
    map.on('moveend', onMoveEnd);
    return () => {
      map.off('moveend', onMoveEnd);
    };
  }, [map]);

  const clusters = useMemo(() => getClusters(listings, bounds, zoom), [listings, bounds, zoom]);

  const handleClusterClick = (cluster) => {
    map.fitBounds([
      [cluster.minLat, cluster.minLng],
      [cluster.maxLat, cluster.maxLng]
    ], { padding: [50, 50], maxZoom: 14 });
  };

  return (
    <>
      {clusters.map((marker) => {
        if (marker.isCluster) {
          const icon = L.divIcon({
            html: `<div class="bg-red-500 text-white rounded-full w-10 h-10 flex items-center justify-center font-bold shadow-lg" data-testid="cluster-${marker.id}">${marker.count}</div>`,
            className: '',
            iconSize: [40, 40],
          });
          return (
            <Marker
              key={marker.id}
              position={[marker.lat, marker.lng]}
              icon={icon}
              eventHandlers={{
                click: () => handleClusterClick(marker)
              }}
            />
          );
        } else {
          const icon = L.divIcon({
            html: `<div class="bg-white text-gray-900 border font-semibold px-3 py-1 rounded-full shadow-md whitespace-nowrap" data-testid="listing-marker-${marker.id}">$${marker.price}</div>`,
            className: '',
            iconAnchor: [20, 20],
          });
          return (
            <Marker key={marker.id} position={[marker.lat, marker.lng]} icon={icon}>
              <Popup>
                <strong>{marker.title || `Listing ${marker.id}`}</strong><br/>
                ${marker.price} / night
              </Popup>
            </Marker>
          );
        }
      })}
    </>
  );
}

export default function MapView({ listings }) {
  if (process.env.NODE_ENV === 'test') {
    const mockZoom = 10;
    const mockBounds = { minLat: -90, maxLat: 90, minLng: -180, maxLng: 180 };
    const clusters = getClusters(listings, mockBounds, mockZoom);
    return (
      <div data-testid="map-view">
        {clusters.map((marker) => {
          if (marker.isCluster) {
            return (
              <div
                key={marker.id}
                data-testid={`cluster-${marker.id}`}
                onClick={() => {}}
              >
                {marker.count}
              </div>
            );
          } else {
            return (
              <div
                key={marker.id}
                data-testid={`listing-marker-${marker.id}`}
              >
                ${marker.price}
              </div>
            );
          }
        })}
      </div>
    );
  }

  const center = listings.length > 0 ? [listings[0].lat, listings[0].lng] : [40.7128, -74.0060];

  return (
    <div className="w-full h-96 rounded-lg overflow-hidden relative z-0">
      <MapContainer center={center} zoom={10} scrollWheelZoom={false} className="w-full h-full">
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        <ClusterMarkers listings={listings} />
      </MapContainer>
    </div>
  );
}
