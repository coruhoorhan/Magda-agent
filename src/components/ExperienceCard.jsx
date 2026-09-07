import React from 'react';

export default function ExperienceCard({ experience }) {
  if (!experience) return null;

  return (
    <div className="group cursor-pointer" data-testid={`exp-card-${experience.id || 'new'}`}>
      <div className="relative aspect-[3/4] overflow-hidden rounded-xl bg-gray-200 dark:bg-gray-800 mb-3">
        <img
          src={experience.imageUrl || 'https://placehold.co/400x533?text=Experience'}
          srcSet={`${experience.imageUrl || 'https://placehold.co/300x400?text=Experience'} 300w, ${experience.imageUrl || 'https://placehold.co/600x800?text=Experience'} 600w`}
          sizes="(max-width: 600px) 100vw, (max-width: 900px) 50vw, 33vw"
          alt={experience.title}
          loading="lazy"
          className="object-cover w-full h-full transition duration-300 group-hover:scale-105"
        />
      </div>
      <div className="flex items-center gap-1 text-sm text-gray-700 dark:text-gray-300 mb-1">
        <svg className="w-3 h-3 fill-current" viewBox="0 0 24 24"><path d="M12 17.27L18.18 21l-1.64-7.03L22 9.24l-7.19-.61L12 2 9.19 8.63 2 9.24l5.46 4.73L5.82 21z"></path></svg>
        <span className="font-semibold">{experience.rating || '5.0'}</span>
        <span className="text-gray-400">({experience.reviews || 0}) · {experience.location || 'Location'}</span>
      </div>
      <h3 className="text-gray-900 dark:text-white line-clamp-2 leading-tight">{experience.title}</h3>
      <div className="mt-1 font-semibold text-gray-900 dark:text-white">
        From ${experience.price} <span className="font-normal">/ person</span>
      </div>
    </div>
  );
}
