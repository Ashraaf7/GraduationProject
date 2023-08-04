using System;
using System.Collections.Generic;
using System.Data;
using System.Data.Entity;
using System.Linq;
using System.Net;
using System.Web;
using System.Web.Mvc;
using CourseApp1.data;

namespace CourseApp1.Controllers
{
    public class RoadmapsController : Controller
    {
        private courses_dbEntities db = new courses_dbEntities();

        // GET: Roadmaps
        public ActionResult Index()
        {
            return View(db.Roadmaps.ToList());
        }
        public ActionResult RoadmapCourses(int? id)
        {
            if (id == null)
            {
                return new HttpStatusCodeResult(HttpStatusCode.BadRequest);
            }
            var roadmap_courses=db.Roadmap_courses.Include(r=>r.Roadmap).Where(c => c.RoadmapId == id).OrderBy(r=>r.Order).ToList();
            ViewBag.RoadmapId = id;
            string roadmapname = db.Roadmaps.Where(c => c.Id == id).Select(c => c.Name).FirstOrDefault();
            ViewBag.RoadName = roadmapname;
            return View(roadmap_courses);
        }

       
        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                db.Dispose();
            }
            base.Dispose(disposing);
        }
    }
}
