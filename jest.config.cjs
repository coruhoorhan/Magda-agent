module.exports = {
  testEnvironment: "jsdom",
  transform: {
    "^.+\\.[t|j]sx?$": "babel-jest",
  },
  moduleNameMapper: {
    "\\.(css|sass)$": "identity-obj-proxy",
  },
  transformIgnorePatterns: [
    "node_modules/(?!(react-leaflet|@react-leaflet|leaflet)/)"
  ],
  setupFilesAfterEnv: ['<rootDir>/tests/setupTests.js']
};
