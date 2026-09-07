import React, { createContext, useContext, useEffect, useState } from 'react';

const PushNotificationContext = createContext({
  isSupported: false,
  isSubscribed: false,
  subscribe: async () => {},
  unsubscribe: async () => {},
});

export const usePushNotifications = () => useContext(PushNotificationContext);

// Base64Url to Uint8Array helper
function urlBase64ToUint8Array(base64String) {
  const padding = '='.repeat((4 - base64String.length % 4) % 4);
  const base64 = (base64String + padding)
    .replace(/\-/g, '+')
    .replace(/_/g, '/');

  const rawData = window.atob(base64);
  const outputArray = new Uint8Array(rawData.length);

  for (let i = 0; i < rawData.length; ++i) {
    outputArray[i] = rawData.charCodeAt(i);
  }
  return outputArray;
}



export function PushNotificationProvider({ children, currentUser }) {
  const [isSupported, setIsSupported] = useState(false);
  const [isSubscribed, setIsSubscribed] = useState(false);
  const [registration, setRegistration] = useState(null);

  useEffect(() => {
    if ('serviceWorker' in navigator && 'PushManager' in window) {
      setIsSupported(true);
      navigator.serviceWorker.register('/service-worker.js')
        .then(reg => {
          setRegistration(reg);
          reg.pushManager.getSubscription().then(sub => {
            setIsSubscribed(!!sub);
          });
        })
        .catch(err => console.error("Service worker registration failed", err));
    }
  }, []);

  const subscribe = async () => {
    if (!registration || !currentUser) return;
    try {
      // Since PUBLIC_VAPID_KEY must match the backend, we should ideally fetch it from backend.
      // For this clone without an endpoint, if we generate dynamically on backend,
      // the frontend won't know the generated key unless we expose it.
      // Let's add a quick fetch to get the public key from the backend.
      const vapidRes = await fetch('/api/notifications/vapid-public-key');
      const { publicKey } = await vapidRes.json();

      const subscription = await registration.pushManager.subscribe({
        userVisibleOnly: true,
        applicationServerKey: urlBase64ToUint8Array(publicKey)
      });


      const response = await fetch('/api/notifications/subscribe', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(subscription)
      });

      if (response.ok) {
        setIsSubscribed(true);
      } else {
        console.error("Failed to save subscription to server");
      }
    } catch (err) {
      console.error("Failed to subscribe to push notifications", err);
    }
  };

  const unsubscribe = async () => {
    if (!registration || !currentUser) return;
    try {
      const subscription = await registration.pushManager.getSubscription();
      if (subscription) {
        await subscription.unsubscribe();
        await fetch('/api/notifications/unsubscribe', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ endpoint: subscription.endpoint })
        });
        setIsSubscribed(false);
      }
    } catch (err) {
      console.error("Failed to unsubscribe from push notifications", err);
    }
  };

  return (
    <PushNotificationContext.Provider value={{ isSupported, isSubscribed, subscribe, unsubscribe }}>
      {children}
    </PushNotificationContext.Provider>
  );
}
