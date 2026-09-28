-- Ce script sert à fusionner les géometries des différentes couches de la BDTOPO et OCSGE qui ne peuvent pas être des
-- zones humides : l'eau libre et les surfaces baties/imperméabilisées.

-- La méthode consiste à utiliser les mêmes filtres sur la BDTOPO que REAUZOH (mask_surfaces_impermeables_bd_topo.py)
-- à l'exeception des réseaus ferrés et routiers : pour les tronçons routiers, on fait un buffer à la largeur du champ buffer.
-- pour les réseaux ferrés, on fait un buffer de 1.435 mètres, qui est la largeur normale selon les spécifications de la bdtopo
-- Cf https://bdtopoexplorer.ign.fr/troncon_de_voie_ferree#attribute_value_919
-- On ajoute également les plans d'eau, qui ne sont pas intégrés dans REAUZOH;

DROP TABLE bdtopo_mask;
CREATE TABLE bdtopo_mask AS(
SELECT geom
	FROM public."N_BATIMENT_TOPO_031"
	WHERE "ETAT"='En service'

UNION ALL

SELECT geom
	FROM public."N_EQUIPEMENT_DE_TRANSPORT_TOPO_031"
	WHERE "ETAT" = 'En service' AND "FICTIF" = 'Non' AND "NAT_DETAIL" NOT IN ('Parking souterrain', 'Parking couvert')

UNION ALL

SELECT geom
	FROM public."N_PISTE_D_AERODROME_TOPO_031"
	WHERE "NATURE" = 'Piste en dur' AND "ETAT" = 'En service'

UNION ALL

SELECT geom 
	FROM public."N_TERRAIN_DE_SPORT_TOPO_031"
	WHERE "ETAT"='En service'

UNION ALL

SELECT geom
	FROM public."N_CIMETIERE_TOPO_031"
	WHERE "ETAT"='En service'

UNION ALL

SELECT geom 
	FROM public."N_SURFACE_HYDROGRAPHIQUE_TOPO_031"
	WHERE PERSISTANC = 'Permanent' AND ETAT = 'En service' AND POS_SOL = '0'

UNION ALL

SELECT geom 
	FROM public."N_PLAN_D_EAU_TOPO_031"

UNION ALL

SELECT ST_Buffer(geom, largeur)
	FROM public."N_TRONCON_DE_ROUTE_TOPO_031"
	WHERE largeur IS NOT NULL

UNION ALL

SELECT ST_Buffer(geom, 1.435)
	FROM public."N_TRONCON_DE_VOIE_FERREE_TOPO_031" WHERE etat='En service'
);

CREATE INDEX bdtopo_mask_gist ON bdtopo_mask USING GIST (geom);