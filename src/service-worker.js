self.addEventListener('push', function(event) {
  if (event.data) {
    try {
      const data = event.data.json();
      const title = data.title || "New Notification";
      const options = {
        body: data.body || "You have a new message.",
        icon: '/vite.svg',
        badge: '/vite.svg'
      };
      event.waitUntil(self.registration.showNotification(title, options));
    } catch(err) {
      console.error("Error parsing push notification data", err);
    }
  }
});

self.addEventListener('notificationclick', function(event) {
  event.notification.close();
  event.waitUntil(
    clients.matchAll({ type: 'window' }).then(windowClients => {
      if (windowClients.length > 0) {
        windowClients[0].focus();
      } else {
        clients.openWindow('/');
      }
    })
  );
});
