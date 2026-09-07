import webpush from "web-push";
import * as db from "./db.js";
import { randomUUID } from "crypto";


// Fallback logic for vapid keys, it must be generated once or provided by environment
// For this clone, if there is no env variable, generate temporary ones or keep missing to fail fast instead of hardcoding private key
let privateVapidKey = process.env.VAPID_PRIVATE_KEY;
let publicVapidKey = process.env.VAPID_PUBLIC_KEY;

if (!privateVapidKey || !publicVapidKey) {
  const generated = webpush.generateVAPIDKeys();
  publicVapidKey = generated.publicKey;
  privateVapidKey = generated.privateKey;
}

webpush.setVapidDetails(
  "mailto:test@example.com",
  publicVapidKey,
  privateVapidKey
);



export function getPublicKey() {
  return publicVapidKey;
}

export function saveSubscription(userId, subscription) {
  const { endpoint, keys } = subscription;
  const subId = randomUUID();
  db.insertPushSubscription({
    id: subId,
    userId,
    endpoint,
    keysP: keys.p256dh,
    keysAuth: keys.auth,
  });
  return subId;
}

export function removeSubscription(userId, endpoint) {
  db.removePushSubscription(userId, endpoint);
}

export async function sendNotification(userId, payload) {
  const subscriptions = db.getPushSubscriptionsForUser(userId);
  if (!subscriptions || subscriptions.length === 0) {
    return;
  }

  const payloadString = JSON.stringify(payload);

  const promises = subscriptions.map((sub) => {
    const pushSubscription = {
      endpoint: sub.endpoint,
      keys: {
        p256dh: sub.keysP,
        auth: sub.keysAuth,
      },
    };

    return webpush.sendNotification(pushSubscription, payloadString).catch((err) => {
      console.error("Error sending notification, removing subscription", err);
      if (err.statusCode === 404 || err.statusCode === 410) {
        db.removePushSubscription(userId, sub.endpoint);
      }
    });
  });

  await Promise.allSettled(promises);
}
