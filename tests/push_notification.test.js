import { describe, it, expect, vi } from 'vitest';
import * as pushEngine from '../src/lib/pushNotificationEngine.js';

describe('Push Notification Engine', () => {
  it('should generate or provide a public VAPID key', () => {
    const key = pushEngine.getPublicKey();
    expect(key).toBeDefined();
    expect(typeof key).toBe('string');
  });
});
