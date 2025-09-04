"use client";

import { useCopilotAction } from "@copilotkit/react-core";

export function LocationAction() {
  useCopilotAction({
    name: "get_user_location",
    description:
      "Get the user's current location from the browser. Always use this before asking for location.",
    available: "remote",
    parameters: [],
    handler: async () => {
      return new Promise((resolve, reject) => {
        if (!navigator.geolocation) {
          reject("Geolocation is not supported by the browser");
        }
        const options = {
          enableHighAccuracy: true,
          timeout: 15000, // 15 seconds of timeout
          maximumAge: 60000, // 1 minute cache TTL
        };
        navigator.geolocation.getCurrentPosition(
          (position) => {
            resolve({
              latitude: position.coords.latitude,
              longitude: position.coords.longitude,
            });
          },
          (error) => {
            reject(`Error getting location: ${error.message}`);
          },
          options
        );
      });
    },
  });
  return null;
}
